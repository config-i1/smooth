#### Internal helper functions shared across ADAM-family models ####
# These are extracted from parametersChecker() to allow reuse by om() and
# other future functions without duplicating logic.

#### Data preparation ####
#' @keywords internal
adam_checkData <- function(data, lags, h, holdout, yName, modelDo, formulaToUse) {
    responseName <- yName

    # Extract from sim/Mdata objects
    if(is.adam.sim(data) || is.smooth.sim(data)){
        data <- data$data
        lags <- frequency(data)
    }
    else if(inherits(data,"Mdata")){
        h <- data$h
        holdout <- TRUE
        if(modelDo!="use"){
            lags <- frequency(data$x)
        }
        data <- ts(c(data$x,data$xx),start=start(data$x),frequency=frequency(data$x))
    }

    # Extract index
    ### tsibble has its own index function, so shit happens because of it...
    if(inherits(data,"tbl_ts")){
        yIndex <- data[[1]]
        if(any(duplicated(yIndex))){
            warning(paste0("You have duplicated time stamps in the variable ",yName,
                           ". I will refactor this."),call.=FALSE)
            yIndex <- yIndex[1] + c(1:length(data[[1]])) * diff(tail(yIndex,2))
        }
    }
    else{
        yIndex <- try(time(data),silent=TRUE)
        if(inherits(yIndex,"try-error")){
            if(!is.data.frame(data) && !is.null(dim(data))){
                yIndex <- as.POSIXct(rownames(data))
            }
            else if(is.data.frame(data)){
                yIndex <- c(1:nrow(data))
            }
            else{
                yIndex <- c(1:length(data))
            }
        }
    }
    yClasses <- class(data)

    # Multi-column data: extract response and xreg
    if(!is.null(ncol(data)) && ncol(data)>1){
        xregData <- data
        if(inherits(data,"tbl_df") || inherits(data,"tbl")){
            data <- as.data.frame(data)
        }
        if(!is.null(formulaToUse)){
            responseName <- all.vars(formulaToUse)[1]
            y <- data[,responseName]
        }
        else{
            responseName <- colnames(xregData)[1]
            if(inherits(data,"tbl_ts")){
                y <- data$value
            }
            else if(inherits(data,"data.table") || inherits(data,"data.frame")){
                y <- data[[1]]
            }
            else if(inherits(data,"zoo")){
                if(ncol(data)>1){
                    xregData <- as.data.frame(data)
                }
                y <- zoo(data[,1],order.by=time(data))
            }
            else{
                y <- data[,1]
            }
        }
        yIndex <- try(time(y),silent=TRUE)
        if(inherits(yIndex,"try-error")){
            if(!is.null(dim(data))){
                yIndex <- try(as.POSIXct(rownames(data)),silent=TRUE)
                if(inherits(yIndex,"try-error")){
                    yIndex <- c(1:nrow(data))
                }
            }
            else{
                yIndex <- c(1:length(y))
            }
        }
        else{
            yClasses <- class(y)
        }
    }
    else{
        xregData <- NULL
        if(!is.null(ncol(data)) && !is.null(colnames(data)[1])){
            responseName <- colnames(data)[1]
            y <- data[,1]
        }
        else{
            y <- data
        }
    }

    responseName <- make.names(responseName)

    obsAll <- length(y) + (1 - holdout)*h
    obsInSample <- length(y) - holdout*h

    if(obsInSample<=0){
        stop("The number of in-sample observations is not positive. Cannot do anything.",
             call.=FALSE)
    }

    # Interpolate NAs using fourier + polynomials
    yNAValues <- is.na(y)
    if(any(yNAValues)){
        warning("Data contains NAs. The values will be ignored during the model construction.",
                call.=FALSE)
        lagMax <- max(max(lags), 10)
        X <- cbind(1, poly(c(1:obsAll), degree=min(max(trunc(obsAll/10),1),5)),
                   sinpi(matrix(c(1:obsAll)*rep(c(1:lagMax),each=obsAll)/lagMax,
                                ncol=lagMax)))
        if(any(y[!yNAValues]<=0)){
            lmFit <- .lm.fit(X[!yNAValues,,drop=FALSE], matrix(y[!yNAValues],ncol=1))
            y[yNAValues] <- (X %*% coef(lmFit))[yNAValues]
        }
        else{
            lmFit <- .lm.fit(X[!yNAValues,,drop=FALSE], matrix(log(y[!yNAValues]),ncol=1))
            y[yNAValues] <- exp(X %*% coef(lmFit))[yNAValues]
        }
        if(!is.null(xregData)){
            xregData[yNAValues,responseName] <- y[yNAValues]
        }
        rm(X)
        if(obsInSample>10000){
            gc(verbose=FALSE)
        }
    }

    # Determine ts class
    if(all(yClasses=="integer") || all(yClasses=="numeric") ||
       all(yClasses=="data.frame") || all(yClasses=="matrix")){
        if(any(class(yIndex) %in% c("POSIXct","Date"))){
            yClasses <- "zoo"
        }
        else{
            yClasses <- "ts"
        }
    }
    yFrequency <- frequency(y)
    yStart <- yIndex[1]
    yInSample <- matrix(y[1:obsInSample],ncol=1)
    if(holdout){
        yForecastStart <- yIndex[obsInSample+1]
        yHoldout <- matrix(y[-c(1:obsInSample)],ncol=1)
        yForecastIndex <- yIndex[-c(1:obsInSample)]
        yInSampleIndex <- yIndex[c(1:obsInSample)]
        yIndexAll <- yIndex
    }
    else{
        yInSampleIndex <- yIndex
        if(any(yClasses=="ts")){
            yIndexDiff <- deltat(yIndex)
            yForecastIndex <- yIndex[obsInSample]+yIndexDiff*c(1:max(h,1))
        }
        else{
            yIndexDiff <- diff(tail(yIndex,2))
            yForecastIndex <- yIndex[obsInSample]+yIndexDiff*c(1:max(h,1))
        }
        yForecastStart <- yIndex[obsInSample]+yIndexDiff
        yHoldout <- NULL
        yIndexAll <- c(yIndex,yForecastIndex)
    }

    if(!is.numeric(yInSample)){
        stop("The provided data is not numeric! Can't construct any model!", call.=FALSE)
    }

    # Add trend variable to xreg if requested via formula but not present
    if(!is.null(formulaToUse) &&
       any(all.vars(formulaToUse)=="trend") && all(colnames(xregData)!="trend")){
        if(!is.null(xregData)){
            xregData <- cbind(xregData,trend=c(1:obsAll))
        }
        else{
            xregData <- cbind(y=y,trend=c(1:obsAll))
        }
    }

    parametersNumber <- matrix(0,2,5,
                               dimnames=list(c("Estimated","Provided"),
                                             c("nParamInternal","nParamXreg",
                                               "nParamOccurrence","nParamScale","nParamAll")))

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
    ))
}

#### Optimiser / ellipsis parameter processing ####
#' @keywords internal
adam_checkOptimizer <- function(ellipsis, loss, distribution, initialType, lags, arimaModel) {
    if(is.null(ellipsis$maxeval)){
        maxeval <- NULL
        if(any(lags>24) && arimaModel && any(initialType==c("optimal","two-stage"))){
            warning(paste0("The estimation of ARIMA model with initial='optimal' on",
                           " high frequency data might take more time to converge.",
                           " Consider setting maxeval to a higher value (e.g.",
                           " maxeval=10000) or using initial='backcasting'."),
                    call.=FALSE, immediate.=TRUE)
        }
    }
    else{
        maxeval <- ellipsis$maxeval
    }
    maxtime <- if(is.null(ellipsis$maxtime)) -1 else ellipsis$maxtime
    xtol_rel <- if(is.null(ellipsis$xtol_rel)) 1E-6 else ellipsis$xtol_rel
    xtol_abs <- if(is.null(ellipsis$xtol_abs)) 1E-8 else ellipsis$xtol_abs
    ftol_rel <- if(is.null(ellipsis$ftol_rel)) 1E-8 else ellipsis$ftol_rel
    ftol_abs <- if(is.null(ellipsis$ftol_abs)) 0 else ellipsis$ftol_abs
    algorithm <- if(is.null(ellipsis$algorithm)) "NLOPT_LN_NELDERMEAD" else ellipsis$algorithm
    print_level <- if(is.null(ellipsis$print_level)) 0 else ellipsis$print_level
    lb <- if(is.null(ellipsis$lb)) NULL else ellipsis$lb
    ub <- if(is.null(ellipsis$ub)) NULL else ellipsis$ub
    B  <- if(is.null(ellipsis$B))  NULL else ellipsis$B

    lambda <- other <- NULL
    otherParameterEstimate <- FALSE

    if(any(loss==c("LASSO","RIDGE"))){
        if(is.null(ellipsis$lambda)){
            warning("You have not provided lambda parameter. I will set it to zero.", call.=FALSE)
            lambda <- 0
        }
        else{
            lambda <- ellipsis$lambda
        }
    }

    if(distribution=="dalaplace"){
        if(is.null(ellipsis$alpha)){
            other <- 0.5
            otherParameterEstimate <- TRUE
        }
        else{
            other <- ellipsis$alpha
            otherParameterEstimate <- FALSE
        }
        names(other) <- "alpha"
    }
    else if(any(distribution==c("dgnorm","dlgnorm"))){
        if(is.null(ellipsis$shape)){
            other <- 2
            otherParameterEstimate <- TRUE
        }
        else{
            other <- ellipsis$shape
            otherParameterEstimate <- FALSE
        }
        names(other) <- "shape"
    }
    else if(distribution=="dt"){
        if(is.null(ellipsis$nu)){
            other <- 2
            otherParameterEstimate <- TRUE
        }
        else{
            other <- ellipsis$nu
            otherParameterEstimate <- FALSE
        }
        names(other) <- "nu"
    }

    if(is.null(ellipsis$nIterations)){
        nIterations <- 1
        if(any(initialType==c("complete","backcasting"))){
            nIterations[] <- 2
        }
    }
    else{
        nIterations <- ellipsis$nIterations
    }

    smoother <- if(is.null(ellipsis$smoother)) "global" else ellipsis$smoother
    FI <- if(is.null(ellipsis$FI)) FALSE else ellipsis$FI
    stepSize <- if(is.null(ellipsis$stepSize)) .Machine$double.eps^(1/4) else ellipsis$stepSize
    dfForBack <- if(is.null(ellipsis$dfForBack)) FALSE else ellipsis$dfForBack

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
    ))
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
        modelIsTrendy <- Ttype != "N"
        if(modelIsTrendy){
            lagsModel <- matrix(c(1, 1), ncol=1)
            componentsNamesETS <- c("level", "trend")
        }
        else{
            lagsModel <- matrix(c(1), ncol=1)
            componentsNamesETS <- c("level")
        }
        modelIsSeasonal <- Stype != "N"
        if(modelIsSeasonal){
            lagsModel <- matrix(c(lagsModel, lagsModelSeasonal), ncol=1)
            componentsNumberETSSeasonal <- length(lagsModelSeasonal)
            if(componentsNumberETSSeasonal > 1){
                componentsNamesETS <- c(componentsNamesETS,
                                        paste0("seasonal", c(1:componentsNumberETSSeasonal)))
            }
            else{
                componentsNamesETS <- c(componentsNamesETS, "seasonal")
            }
        }
        else{
            componentsNumberETSSeasonal <- 0
        }
        lagsModelAll <- lagsModel
        componentsNumberETS <- length(lagsModel)
    }
    else{
        modelIsTrendy <- modelIsSeasonal <- FALSE
        componentsNumberETS <- componentsNumberETSSeasonal <- 0
        componentsNamesETS <- NULL
        lagsModelAll <- lagsModel <- NULL
    }

    if(arimaModel){
        lagsModelAll <- matrix(c(lagsModel, lagsModelARIMA), ncol=1)
    }

    if(constantRequired){
        lagsModelAll <- matrix(c(lagsModelAll, 1), ncol=1)
    }

    if(xregModel){
        lagsModelAll <- matrix(c(lagsModelAll, rep(1, xregNumber)), ncol=1)
    }

    lagsModelMax <- max(lagsModelAll)
    obsStates <- obsInSample + lagsModelMax

    adamProfiles <- adamProfileCreator(lagsModelAll, lagsModelMax, obsAll,
                                       lags=lags, yIndex=yIndexAll, yClasses=yClasses)
    if(profilesRecentProvided){
        profilesRecentTable <- profilesRecentTable[, 1:lagsModelMax, drop=FALSE]
    }
    else{
        profilesRecentTable <- adamProfiles$recent
    }
    indexLookupTable <- adamProfiles$lookup

    componentsNumberETSNonSeasonal <- componentsNumberETS - componentsNumberETSSeasonal
    adamCpp <- new(adamCore,
                   lagsModelAll, Etype, Ttype, Stype,
                   componentsNumberETSNonSeasonal,
                   componentsNumberETSSeasonal,
                   componentsNumberETS, componentsNumberARIMA,
                   xregNumber, length(lagsModelAll),
                   constantRequired, adamETS)

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
    ))
}

#' @keywords internal
adam_creator <- function(etsModel, Etype, Ttype, Stype, modelIsTrendy, modelIsSeasonal,
                         lags, lagsModel, lagsModelARIMA, lagsModelAll, lagsModelMax,
                         profilesRecentTable=NULL, profilesRecentProvided=FALSE,
                         obsStates, obsInSample, obsAll,
                         componentsNumberETS, componentsNumberETSSeasonal,
                         componentsNamesETS, otLogical, yInSample,
                         persistence, persistenceEstimate,
                         persistenceLevel, persistenceLevelEstimate,
                         persistenceTrend, persistenceTrendEstimate,
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

    nComponents <- componentsNumberETS+componentsNumberARIMA+xregNumber+constantRequired
    componentNames <- c(componentsNamesETS, componentsNamesARIMA, xregNames, constantName)

    matVt <- matrix(NA, nComponents, obsStates,
                    dimnames=list(componentNames, NULL))

    matWt <- matrix(1, obsAll, nComponents,
                    dimnames=list(NULL, componentNames))

    if(xregModel){
        matWt[,componentsNumberETS+componentsNumberARIMA+1:xregNumber] <- xregData
    }

    matF <- diag(nComponents)

    vecG <- matrix(0, nComponents, 1,
                   dimnames=list(componentNames, NULL))

    j <- 0
    if(etsModel){
        j <- j+1
        rownames(vecG)[j] <- "alpha"
        if(!persistenceLevelEstimate){
            vecG[j,] <- persistenceLevel
        }
        if(modelIsTrendy){
            j <- j+1
            rownames(vecG)[j] <- "beta"
            if(!persistenceTrendEstimate){
                vecG[j,] <- persistenceTrend
            }
        }
        if(modelIsSeasonal){
            if(!all(persistenceSeasonalEstimate)){
                vecG[j+which(!persistenceSeasonalEstimate),] <- persistenceSeasonal
            }
            if(componentsNumberETSSeasonal>1){
                rownames(vecG)[j+c(1:componentsNumberETSSeasonal)] <-
                    paste0("gamma", c(1:componentsNumberETSSeasonal))
            }
            else{
                rownames(vecG)[j+1] <- "gamma"
            }
            j <- j+componentsNumberETSSeasonal
        }
    }

    if(arimaModel){
        matF[j+1:componentsNumberARIMA,j+1:componentsNumberARIMA] <- 0
        if(componentsNumberARIMA>1){
            rownames(vecG)[j+1:componentsNumberARIMA] <- paste0("psi",c(1:componentsNumberARIMA))
        }
        else{
            rownames(vecG)[j+1:componentsNumberARIMA] <- "psi"
        }
        j <- j+componentsNumberARIMA
    }

    if(!arimaModel && constantRequired){
        matF[1,ncol(matF)] <- 1
    }

    if(xregModel){
        if(persistenceXregProvided && !persistenceXregEstimate){
            vecG[j+1:xregNumber,] <- persistenceXreg
        }
        rownames(vecG)[j+1:xregNumber] <- paste0("delta",xregParametersPersistence)
    }

    if(etsModel && modelIsTrendy){
        matF[1,2] <- phi
        matF[2,2] <- phi
        matWt[,2] <- phi
    }

    if(arimaModel && (!arEstimate && !maEstimate)){
        arimaPolynomials <- lapply(
            adamCpp$polynomialise(0, arOrders, iOrders, maOrders,
                                  arEstimate, maEstimate, armaParameters, lags),
            as.vector)
        if(nrow(nonZeroARI)>0){
            matF[componentsNumberETS+nonZeroARI[,2],componentsNumberETS+nonZeroARI[,2]] <-
                -arimaPolynomials$ariPolynomial[nonZeroARI[,1]]
        }
        if(nrow(nonZeroARI)>0){
            vecG[componentsNumberETS+nonZeroARI[,2]] <-
                -arimaPolynomials$ariPolynomial[nonZeroARI[,1]]
        }
        if(nrow(nonZeroMA)>0){
            vecG[componentsNumberETS+nonZeroMA[,2]] <- vecG[componentsNumberETS+nonZeroMA[,2]] +
                arimaPolynomials$maPolynomial[nonZeroMA[,1]]
        }
    }
    else{
        arimaPolynomials <- NULL
    }

    if(!profilesRecentProvided){
        if(etsModel){
            if(initialEstimate){
                if(modelIsSeasonal){
                    yDecompositionAdditive <- msdecompose(yInSample, lags=lags[lags!=1],
                                                          type="additive",
                                                          smoother=smoother)
                    if(any(c(Etype,Ttype,Stype)=="M")){
                        yDecompositionMultiplicative <- msdecompose(yInSample, lags=lags[lags!=1],
                                                                    type="multiplicative",
                                                                    smoother=smoother)
                    }
                    decompositionType <- c("additive","multiplicative")[any(c(Etype,Stype)=="M")+1]
                    yDecomposition <- switch(decompositionType,
                                             "additive"=yDecompositionAdditive,
                                             "multiplicative"=yDecompositionMultiplicative)
                    j <- 1
                    if(initialLevelEstimate){
                        if(modelIsTrendy){
                            matVt[j,1:lagsModelMax] <- switch(Ttype,
                                "M"=yDecompositionMultiplicative$initial$nonseasonal[1],
                                yDecompositionAdditive$initial$nonseasonal[1])
                        }
                        else{
                            matVt[j,1:lagsModelMax] <- mean(yInSample[otLogical])
                        }
                        if(xregModel){
                            if(Etype=="A"){
                                matVt[j,1:lagsModelMax] <- matVt[j,1:lagsModelMax] -
                                    as.vector(xregModelInitials[[1]]$initialXreg %*% xregData[1,])
                            }
                            else{
                                matVt[j,1:lagsModelMax] <- matVt[j,1:lagsModelMax] /
                                    as.vector(exp(xregModelInitials[[2]]$initialXreg %*%
                                                      xregData[1,]))
                            }
                        }
                    }
                    else{
                        matVt[j,1:lagsModelMax] <- initialLevel
                    }
                    j <- j+1
                    if(modelIsTrendy){
                        if(initialTrendEstimate){
                            if(Ttype=="A"){
                                matVt[j,1:lagsModelMax] <-
                                    yDecompositionAdditive$initial$nonseasonal[2]
                                if(Stype=="M"){
                                    if(matVt[j,1]<0 &&
                                       abs(matVt[j,1])>min(abs(yInSample[otLogical]))){
                                        matVt[j,1:lagsModelMax] <- 0
                                    }
                                }
                            }
                            else if(Ttype=="M"){
                                matVt[j,1:lagsModelMax] <-
                                    yDecompositionMultiplicative$initial$nonseasonal[2]
                                if(any(matVt[1,1:lagsModelMax]<0)){
                                    matVt[1,1:lagsModelMax] <- yInSample[otLogical][1]
                                }
                            }
                        }
                        else{
                            matVt[j,1:lagsModelMax] <- initialTrend
                        }
                        j <- j+1
                    }
                    if(all(c(Etype,Stype)=="A") || all(c(Etype,Stype)=="M") ||
                       (Etype=="A" & Stype=="M")){
                        for(i in 1:componentsNumberETSSeasonal){
                            if(initialSeasonalEstimate[i]){
                                matVt[i+j-1,1:lagsModel[i+j-1]] <-
                                    yDecomposition$initial$seasonal[[i]]
                                if(Stype=="A"){
                                    matVt[i+j-1,1:lagsModel[i+j-1]] <-
                                        matVt[i+j-1,1:lagsModel[i+j-1]] -
                                        mean(matVt[i+j-1,1:lagsModel[i+j-1]])
                                }
                                else{
                                    matVt[i+j-1,1:lagsModel[i+j-1]] <-
                                        matVt[i+j-1,1:lagsModel[i+j-1]] /
                                        exp(mean(log(matVt[i+j-1,1:lagsModel[i+j-1]])))
                                }
                            }
                            else{
                                matVt[i+j-1,1:lagsModel[i+j-1]] <- initialSeasonal[[i]]
                            }
                        }
                    }
                    else if(Etype=="M" && Stype=="A"){
                        for(i in 1:componentsNumberETSSeasonal){
                            if(initialSeasonalEstimate[i]){
                                matVt[i+j-1,1:lagsModel[i+j-1]] <-
                                    log(yDecomposition$initial$seasonal[[i]]) *
                                    min(yInSample[otLogical])
                                if(Stype=="A"){
                                    matVt[i+j-1,1:lagsModel[i+j-1]] <-
                                        matVt[i+j-1,1:lagsModel[i+j-1]] -
                                        mean(matVt[i+j-1,1:lagsModel[i+j-1]])
                                }
                                else{
                                    matVt[i+j-1,1:lagsModel[i+j-1]] <-
                                        matVt[i+j-1,1:lagsModel[i+j-1]] /
                                        exp(mean(log(matVt[i+j-1,1:lagsModel[i+j-1]])))
                                }
                            }
                            else{
                                matVt[i+j-1,1:lagsModel[i+j-1]] <- initialSeasonal[[i]]
                            }
                        }
                    }
                    if(Etype=="M" && matVt[1,1]<=0){
                        matVt[1,1:lagsModelMax] <- yInSample[1]
                    }
                }
                else{
                    yDecompositionAdditive <- msdecompose(yInSample, lags=1,
                                                          type="additive",
                                                          smoother=smoother)
                    if(any(c(Etype,Ttype)=="M")){
                        yDecompositionMultiplicative <- msdecompose(yInSample, lags=1,
                                                                    type="multiplicative",
                                                                    smoother=smoother)
                    }
                    if(initialLevelEstimate){
                        if(modelIsTrendy){
                            matVt[1,1:lagsModelMax] <- switch(Ttype,
                                "M"=yDecompositionMultiplicative$initial$nonseasonal[1],
                                yDecompositionAdditive$initial$nonseasonal[1])
                        }
                        else{
                            matVt[1,1:lagsModelMax] <- mean(yInSample[otLogical])
                        }
                    }
                    else{
                        matVt[1,1:lagsModelMax] <- initialLevel
                    }
                    if(modelIsTrendy){
                        if(initialTrendEstimate){
                            matVt[2,1:lagsModelMax] <- switch(Ttype,
                                "A"=yDecompositionAdditive$initial$nonseasonal[2],
                                "M"=yDecompositionMultiplicative$initial$nonseasonal[2])
                        }
                        else{
                            matVt[2,1:lagsModelMax] <- initialTrend
                        }
                    }
                    if(Etype=="M" && matVt[1,1]<=0){
                        matVt[1,1:lagsModelMax] <- yInSample[1]
                    }
                }
                if(initialLevelEstimate && Etype=="M" && matVt[1,lagsModelMax]==0){
                    matVt[1,1:lagsModelMax] <- mean(yInSample)
                }
            }
            else if(!initialEstimate && initialType=="provided"){
                j <- 1
                matVt[j,1:lagsModelMax] <- initialLevel
                if(modelIsTrendy){
                    j <- j+1
                    matVt[j,1:lagsModelMax] <- initialTrend
                }
                if(modelIsSeasonal){
                    for(i in 1:componentsNumberETSSeasonal){
                        matVt[j+i,1:lagsModel[j+i]] <- initialSeasonal[[i]]
                    }
                }
                j <- j+componentsNumberETSSeasonal
            }
        }

        if(arimaModel){
            if(initialArimaEstimate){
                matVt[componentsNumberETS+1:componentsNumberARIMA, 1:initialArimaNumber] <-
                    switch(Etype, "A"=0, "M"=1)
                if(any(lags>1) && obsInSample > max(lags)*2){
                    yDecomposition <- tail(msdecompose(yInSample,
                                                       lags=lags[lags!=1],
                                                       type=switch(Etype,
                                                                   "A"="additive",
                                                                   "M"="multiplicative"),
                                                       smoother=smoother)$seasonal, 1)[[1]]
                }
                else if(any(lags>1) && obsInSample <= max(lags)*2){
                    yDecomposition <- yInSample[otLogical][1:obsInSample]
                }
                else{
                    yDecomposition <- switch(Etype,
                                             "A"=mean(diff(yInSample[otLogical])),
                                             "M"=exp(mean(diff(log(yInSample[otLogical])))))
                }
                matVt[componentsNumberETS+componentsNumberARIMA, 1:initialArimaNumber] <-
                    rep(yDecomposition, ceiling(initialArimaNumber/max(lags)))[1:initialArimaNumber]
            }
            else{
                matVt[componentsNumberETS+1:componentsNumberARIMA, 1:initialArimaNumber] <-
                    switch(Etype, "A"=0, "M"=1)
                matVt[componentsNumberETS+componentsNumberARIMA, 1:initialArimaNumber] <-
                    initialArima[1:initialArimaNumber]
            }
        }

        if(xregModel){
            if(Etype=="A" || initialXregProvided || is.null(xregModelInitials[[2]])){
                matVt[componentsNumberETS+componentsNumberARIMA+1:xregNumber, 1:lagsModelMax] <- 0
                matVt[names(xregModelInitials[[1]]$initialXreg), 1:lagsModelMax] <-
                    xregModelInitials[[1]]$initialXreg
            }
            else{
                matVt[componentsNumberETS+componentsNumberARIMA+1:xregNumber, 1:lagsModelMax] <- 0
                matVt[names(xregModelInitials[[2]]$initialXreg), 1:lagsModelMax] <-
                    xregModelInitials[[2]]$initialXreg
            }
        }

        if(constantRequired){
            if(constantEstimate){
                if(sum(iOrders)==0 && !etsModel){
                    matVt[componentsNumberETS+componentsNumberARIMA+xregNumber+1,] <-
                        mean(yInSample[otLogical])
                }
                else{
                    matVt[componentsNumberETS+componentsNumberARIMA+xregNumber+1,] <-
                        switch(Etype,
                               "A"=mean(diff(yInSample[otLogical])),
                               "M"=exp(mean(diff(log(yInSample[otLogical])))))
                }
            }
            else{
                matVt[componentsNumberETS+componentsNumberARIMA+xregNumber+1,] <- constantValue
            }
            if(etsModel && initialLevelEstimate){
                if(Etype=="A"){
                    matVt[1,1:lagsModelMax] <- matVt[1,1:lagsModelMax] -
                        matVt[componentsNumberETS+componentsNumberARIMA+xregNumber+1,1]
                }
                else{
                    matVt[1,1:lagsModelMax] <- matVt[1,1:lagsModelMax] /
                        matVt[componentsNumberETS+componentsNumberARIMA+xregNumber+1,1]
                }
            }
            if(arimaModel && initialArimaEstimate){
                if(Etype=="A"){
                    matVt[componentsNumberETS+nonZeroARI[,2],1:initialArimaNumber] <-
                        matVt[componentsNumberETS+nonZeroARI[,2],1:initialArimaNumber] -
                        matVt[componentsNumberETS+componentsNumberARIMA+xregNumber+1,1]
                }
                else{
                    matVt[componentsNumberETS+nonZeroARI[,2],1:initialArimaNumber] <-
                        matVt[componentsNumberETS+nonZeroARI[,2],1:initialArimaNumber] /
                        matVt[componentsNumberETS+componentsNumberARIMA+xregNumber+1,1]
                }
            }
        }
    }
    else{
        matVt[,1:lagsModelMax] <- profilesRecentTable
    }

    return(list(matVt=matVt, matWt=matWt, matF=matF, vecG=vecG, arimaPolynomials=arimaPolynomials))
}

#' @keywords internal
adam_filler <- function(B,
                        etsModel, Etype, Ttype, Stype, modelIsTrendy, modelIsSeasonal,
                        componentsNumberETS, componentsNumberETSNonSeasonal,
                        componentsNumberETSSeasonal, componentsNumberARIMA,
                        lags, lagsModel, lagsModelMax,
                        matVt, matWt, matF, vecG,
                        persistenceEstimate, persistenceLevelEstimate, persistenceTrendEstimate,
                        persistenceSeasonalEstimate, persistenceXregEstimate,
                        phiEstimate,
                        initialType, initialEstimate,
                        initialLevelEstimate, initialTrendEstimate, initialSeasonalEstimate,
                        initialArimaEstimate, initialXregEstimate,
                        arimaModel, arEstimate, maEstimate, arOrders, iOrders, maOrders,
                        arRequired, maRequired, armaParameters,
                        nonZeroARI, nonZeroMA, arimaPolynomials,
                        xregModel, xregNumber,
                        xregParametersMissing, xregParametersIncluded,
                        xregParametersEstimated, xregParametersPersistence,
                        constantEstimate,
                        adamCpp,
                        constantRequired, initialArimaNumber){

    j <- 0
    # Fill in persistence
    if(persistenceEstimate){
        # Persistence of ETS
        if(etsModel){
            i <- 1
            # alpha
            if(persistenceLevelEstimate){
                j[] <- j+1
                vecG[i] <- B[j]
            }
            # beta
            if(modelIsTrendy){
                i[] <- 2
                if(persistenceTrendEstimate){
                    j[] <- j+1
                    vecG[i] <- B[j]
                }
            }
            # gamma1, gamma2, ...
            if(modelIsSeasonal){
                if(any(persistenceSeasonalEstimate)){
                    vecG[i+which(persistenceSeasonalEstimate)] <-
                        B[j+c(1:sum(persistenceSeasonalEstimate))]
                    j[] <- j+sum(persistenceSeasonalEstimate)
                }
                i[] <- componentsNumberETS
            }
        }

        # Persistence of xreg
        if(xregModel && persistenceXregEstimate){
            xregPersistenceNumber <- max(xregParametersPersistence)
            vecG[j+componentsNumberARIMA+1:length(xregParametersPersistence)] <-
                B[j+1:xregPersistenceNumber][xregParametersPersistence]
            j[] <- j+xregPersistenceNumber
        }
    }

    # Damping parameter
    if(etsModel && phiEstimate){
        j[] <- j+1
        matWt[,2] <- B[j]
        matF[1:2,2] <- B[j]
    }

    # ARMA parameters. This goes before xreg in persistence
    if(arimaModel){
        # Call the function returning ARI and MA polynomials
        arimaPolynomials <- lapply(
            adamCpp$polynomialise(B[j+1:sum(c(arOrders*arEstimate,maOrders*maEstimate))],
                                  arOrders, iOrders, maOrders,
                                  arEstimate, maEstimate, armaParameters, lags),
            as.vector)

        # Fill in the transition matrix
        if(nrow(nonZeroARI)>0){
            matF[componentsNumberETS+nonZeroARI[,2],
                 componentsNumberETS+1:(componentsNumberARIMA+constantRequired)] <-
                -arimaPolynomials$ariPolynomial[nonZeroARI[,1]]
        }
        # Fill in the persistence vector
        if(nrow(nonZeroARI)>0){
            vecG[componentsNumberETS+nonZeroARI[,2]] <-
                -arimaPolynomials$ariPolynomial[nonZeroARI[,1]]
        }
        if(nrow(nonZeroMA)>0){
            vecG[componentsNumberETS+nonZeroMA[,2]] <- vecG[componentsNumberETS+nonZeroMA[,2]] +
                arimaPolynomials$maPolynomial[nonZeroMA[,1]]
        }
        j[] <- j+sum(c(arOrders*arEstimate,maOrders*maEstimate))
    }

    # Initials of ETS if something needs to be estimated
    if(etsModel && all(initialType!=c("complete","backcasting")) && initialEstimate){
        i <- 1
        if(initialLevelEstimate){
            j[] <- j+1
            matVt[i,1:lagsModelMax] <- B[j]
        }
        i[] <- i+1
        if(modelIsTrendy && initialTrendEstimate){
            j[] <- j+1
            matVt[i,1:lagsModelMax] <- B[j]
            i[] <- i+1
        }
        if(modelIsSeasonal && any(initialSeasonalEstimate)){
            for(k in 1:componentsNumberETSSeasonal){
                if(initialSeasonalEstimate[k]){
                    matVt[componentsNumberETSNonSeasonal+k,
                          2:lagsModel[componentsNumberETSNonSeasonal+k]-1] <-
                        B[j+2:(lagsModel[componentsNumberETSNonSeasonal+k])-1]
                    matVt[componentsNumberETSNonSeasonal+k,
                          lagsModel[componentsNumberETSNonSeasonal+k]] <-
                        switch(Stype,
                               "A"=-sum(B[j+2:(lagsModel[componentsNumberETSNonSeasonal+k])-1]),
                               "M"=1/prod(B[j+2:(lagsModel[componentsNumberETSNonSeasonal+k])-1]))
                    j[] <- j+lagsModel[componentsNumberETSNonSeasonal+k]-1
                }
            }
        }
    }

    # Initials of ARIMA
    if(arimaModel){
        if(all(initialType!=c("complete","backcasting")) && initialArimaEstimate){
            matVt[componentsNumberETS+nonZeroARI[,2], 1:initialArimaNumber] <-
                switch(Etype,
                       "A"=arimaPolynomials$ariPolynomial[nonZeroARI[,1]] %*%
                           t(B[j+1:initialArimaNumber]),
                       "M"=exp(arimaPolynomials$ariPolynomial[nonZeroARI[,1]] %*%
                                   t(log(B[j+1:initialArimaNumber]))))
            j[] <- j+initialArimaNumber
        }
        # This is needed in order to propagate initials of ARIMA to all components
        else if(any(c(arEstimate,maEstimate))){
            matVt[componentsNumberETS+nonZeroARI[,2], 1:initialArimaNumber] <-
                switch(Etype,
                       "A"= arimaPolynomials$ariPolynomial[nonZeroARI[,1]] %*%
                           t(matVt[componentsNumberETS+componentsNumberARIMA,
                                   1:initialArimaNumber]),
                       "M"=exp(arimaPolynomials$ariPolynomial[nonZeroARI[,1]] %*%
                                   t(log(matVt[componentsNumberETS+componentsNumberARIMA,
                                               1:initialArimaNumber]))))
        }
    }

    # Initials of the xreg
    if(xregModel && (initialType!="complete") && initialEstimate && initialXregEstimate){
        xregNumberToEstimate <- sum(xregParametersEstimated)
        matVt[componentsNumberETS+componentsNumberARIMA+which(xregParametersEstimated==1),
              1:lagsModelMax] <- B[j+1:xregNumberToEstimate]
        j[] <- j+xregNumberToEstimate
    }

    # Constant
    if(constantEstimate){
        matVt[componentsNumberETS+componentsNumberARIMA+xregNumber+1,] <- B[j+1]
    }

    return(list(matVt=matVt, matWt=matWt, matF=matF, vecG=vecG, arimaPolynomials=arimaPolynomials))
}

#' @keywords internal
adam_initialiser <- function(etsModel, Etype, Ttype, Stype, modelIsTrendy, modelIsSeasonal,
                              componentsNumberETSNonSeasonal, componentsNumberETSSeasonal,
                              componentsNumberETS,
                              lags, lagsModel, lagsModelSeasonal, lagsModelARIMA, lagsModelMax,
                              matVt,
                              persistenceEstimate, persistenceLevelEstimate,
                              persistenceTrendEstimate,
                              persistenceSeasonalEstimate, persistenceXregEstimate,
                              phiEstimate, initialType, initialEstimate,
                              initialLevelEstimate, initialTrendEstimate, initialSeasonalEstimate,
                              initialArimaEstimate, initialXregEstimate,
                              arimaModel, arRequired, maRequired, arEstimate, maEstimate,
                              arOrders, maOrders,
                              componentsNumberARIMA, componentsNamesARIMA, initialArimaNumber,
                              xregModel, xregNumber,
                              xregParametersEstimated, xregParametersPersistence,
                              constantEstimate, constantName, otherParameterEstimate,
                              adamCpp,
                              ets, bounds, yInSample, otLogical, iOrders, armaParameters, other){
    # The vector of logicals for persistence elements
    persistenceEstimateVector <- c(persistenceLevelEstimate,
                                   modelIsTrendy&persistenceTrendEstimate,
                                   modelIsSeasonal&persistenceSeasonalEstimate)

    # The order:
    # Persistence of states and for xreg, phi, AR and MA parameters,
    # initials, initialsARIMA, initials for xreg
    B <- Bl <- Bu <- vector("numeric",
                            # Values of the persistence vector + phi
                            etsModel*(persistenceLevelEstimate +
                                          modelIsTrendy*persistenceTrendEstimate +
                                          modelIsSeasonal*sum(persistenceSeasonalEstimate) +
                                          phiEstimate) +
                                xregModel*persistenceXregEstimate*max(xregParametersPersistence) +
                                # AR and MA values
                                arimaModel*(arEstimate*sum(arOrders)+maEstimate*sum(maOrders)) +
                                # initials of ETS
                                etsModel*all(initialType!=c("complete","backcasting"))*
                                (initialLevelEstimate +
                                     (modelIsTrendy*initialTrendEstimate) +
                                     (modelIsSeasonal*
                                      sum(initialSeasonalEstimate*(lagsModelSeasonal-1)))) +
                                # initials of ARIMA
                                all(initialType!=c("complete","backcasting"))*
                                arimaModel*initialArimaNumber*initialArimaEstimate +
                                # initials of xreg
                                (initialType!="complete")*xregModel*initialXregEstimate*
                                sum(xregParametersEstimated) +
                                constantEstimate + otherParameterEstimate)

    j <- 0
    if(etsModel){
        # Fill in persistence
        if(persistenceEstimate && any(persistenceEstimateVector)){
            if(ets=="conventional" && any(c(Etype,Ttype,Stype)=="M")){
                # A special type of model which is not safe: AAM, MAA, MAM
                if((Etype=="A" && Ttype=="A" && Stype=="M") ||
                   (Etype=="A" && Ttype=="M" && Stype=="A") ||
                   (any(initialType==c("complete","backcasting")) &&
                    ((Etype=="M" && Ttype=="A" && Stype=="A") ||
                     (Etype=="M" && Ttype=="A" && Stype=="M")))){
                    B[1:sum(persistenceEstimateVector)] <-
                        c(0.01,0.005,rep(0.001,componentsNumberETSSeasonal))[
                            which(persistenceEstimateVector)]
                }
                # MMA is the worst. Set everything to zero and see if anything can be done...
                else if((Etype=="M" && Ttype=="M" && Stype=="A")){
                    B[1:sum(persistenceEstimateVector)] <-
                        c(0.01,0.005,rep(0.01,componentsNumberETSSeasonal))[
                            which(persistenceEstimateVector)]
                }
                else if(Etype=="M" && Ttype=="A"){
                    if(any(initialType==c("complete","backcasting"))){
                        B[1:sum(persistenceEstimateVector)] <-
                            c(0.1,0.05,rep(0.3,componentsNumberETSSeasonal))[
                                which(persistenceEstimateVector)]
                    }
                    else{
                        B[1:sum(persistenceEstimateVector)] <-
                            c(0.2,0.01,rep(0.3,componentsNumberETSSeasonal))[
                                which(persistenceEstimateVector)]
                    }
                }
                else if(Etype=="M" && Ttype=="M"){
                    B[1:sum(persistenceEstimateVector)] <-
                        c(0.1,0.05,rep(0.3,componentsNumberETSSeasonal))[
                            which(persistenceEstimateVector)]
                }
                else{
                    B[1:sum(persistenceEstimateVector)] <-
                        c(0.1,0.05,rep(0.3,componentsNumberETSSeasonal))[
                            which(persistenceEstimateVector)]
                }
            }
            else{
                B[1:sum(persistenceEstimateVector)] <-
                    c(0.1, 0.05, rep(0.3, componentsNumberETSSeasonal))[
                        which(persistenceEstimateVector)]
            }
            if(bounds=="usual"){
                Bl[1:sum(persistenceEstimateVector)] <- rep(0, sum(persistenceEstimateVector))
                Bu[1:sum(persistenceEstimateVector)] <- rep(1, sum(persistenceEstimateVector))
            }
            else{
                Bl[1:sum(persistenceEstimateVector)] <- rep(-5, sum(persistenceEstimateVector))
                Bu[1:sum(persistenceEstimateVector)] <- rep(5, sum(persistenceEstimateVector))
            }
            # Names for B
            if(persistenceLevelEstimate){
                j[] <- j+1
                names(B)[j] <- "alpha"
            }
            if(modelIsTrendy && persistenceTrendEstimate){
                j[] <- j+1
                names(B)[j] <- "beta"
            }
            if(modelIsSeasonal && any(persistenceSeasonalEstimate)){
                if(componentsNumberETSSeasonal>1){
                    names(B)[j+c(1:sum(persistenceSeasonalEstimate))] <-
                        paste0("gamma",c(1:componentsNumberETSSeasonal))
                }
                else{
                    names(B)[j+1] <- "gamma"
                }
                j[] <- j+sum(persistenceSeasonalEstimate)
            }
        }
    }

    # Persistence if xreg is provided
    if(xregModel && persistenceXregEstimate){
        xregPersistenceNumber <- max(xregParametersPersistence)
        B[j+1:xregPersistenceNumber] <- rep(switch(Etype,"A"=0.01,"M"=0),xregPersistenceNumber)
        Bl[j+1:xregPersistenceNumber] <- rep(-5, xregPersistenceNumber)
        Bu[j+1:xregPersistenceNumber] <- rep(5, xregPersistenceNumber)
        names(B)[j+1:xregPersistenceNumber] <- paste0("delta",c(1:xregPersistenceNumber))
        j[] <- j+xregPersistenceNumber
    }

    # Damping parameter
    if(etsModel && phiEstimate){
        j[] <- j+1
        B[j] <- 0.95
        names(B)[j] <- "phi"
        Bl[j] <- 0
        Bu[j] <- 1
    }

    # ARIMA parameters (AR / MA)
    if(arimaModel){
        # This index is needed to get the correct polynomials
        k <- j
        # These are filled in lags-wise
        if(any(c(arEstimate,maEstimate))){
            acfValues <- rep(-0.1, maOrders %*% lags)
            pacfValues <- rep(0.1, arOrders %*% lags)
            # If this is ETS + ARIMA model or no differences model, then don't bother with initials
            # The latter does not make sense because of non-stationarity in ACF / PACF
            # Otherwise use ACF / PACF values as starting parameters for ARIMA
            if(!(etsModel || all(iOrders==0))){
                yDifferenced <- yInSample
                # If the model has differences, take them
                if(any(iOrders>0)){
                    for(i in 1:length(iOrders)){
                        if(iOrders[i]>0){
                            yDifferenced <- diff(yDifferenced,lag=lags[i],differences=iOrders[i])
                        }
                    }
                }
                # Do ACF/PACF initialisation only for non-seasonal models
                if(all(lags<=1)){
                    if(maRequired && maEstimate){
                        acfValues[1:min(maOrders %*% lags, length(yDifferenced)-1)] <-
                            acf(yDifferenced, lag.max=max(1,maOrders %*% lags),
                                plot=FALSE)$acf[-1]
                    }
                    if(arRequired && arEstimate){
                        pacfValues[1:min(arOrders %*% lags, length(yDifferenced)-1)] <-
                            pacf(yDifferenced, lag.max=max(1,arOrders %*% lags),
                                 plot=FALSE)$acf
                    }
                }
            }
            for(i in 1:length(lags)){
                if(arRequired && arEstimate && arOrders[i]>0){
                    if(all(!is.nan(pacfValues[c(1:arOrders[i])*lags[i]]))){
                        B[j+c(1:arOrders[i])] <- pacfValues[c(1:arOrders[i])*lags[i]]
                    }
                    else{
                        B[j+c(1:arOrders[i])] <- 0.1
                    }
                    if(sum(B[j+c(1:arOrders[i])])>1){
                        B[j+c(1:arOrders[i])] <- B[j+c(1:arOrders[i])] /
                            sum(B[j+c(1:arOrders[i])]) - 0.01
                    }
                    Bl[j+c(1:arOrders[i])] <- -5
                    Bu[j+c(1:arOrders[i])] <- 5
                    names(B)[j+1:arOrders[i]] <- paste0("phi",1:arOrders[i],"[",lags[i],"]")
                    j[] <- j + arOrders[i]
                }
                if(maRequired && maEstimate && maOrders[i]>0){
                    if(all(!is.nan(acfValues[c(1:maOrders[i])*lags[i]]))){
                        B[j+c(1:maOrders[i])] <- acfValues[c(1:maOrders[i])*lags[i]]
                    }
                    else{
                        B[j+c(1:maOrders[i])] <- 0.1
                    }
                    if(sum(B[j+c(1:maOrders[i])])>1){
                        B[j+c(1:maOrders[i])] <- B[j+c(1:maOrders[i])] /
                            sum(B[j+c(1:maOrders[i])]) - 0.01
                    }
                    Bl[j+c(1:maOrders[i])] <- -5
                    Bu[j+c(1:maOrders[i])] <- 5
                    names(B)[j+1:maOrders[i]] <- paste0("theta",1:maOrders[i],"[",lags[i],"]")
                    j[] <- j + maOrders[i]
                }
            }
        }

        arimaPolynomials <- lapply(
            adamCpp$polynomialise(B[k+1:sum(c(arOrders*arEstimate,maOrders*maEstimate))],
                                  arOrders, iOrders, maOrders,
                                  arEstimate, maEstimate, armaParameters, lags),
            as.vector)
    }

    # Initials
    if(etsModel && all(initialType!=c("complete","backcasting")) && initialEstimate){
        if(initialLevelEstimate){
            j[] <- j+1
            B[j] <- matVt[1,1]
            names(B)[j] <- "level"
            if(Etype=="A"){
                Bl[j] <- -Inf
                Bu[j] <- Inf
            }
            else{
                Bl[j] <- 0
                Bu[j] <- Inf
            }
        }
        if(modelIsTrendy && initialTrendEstimate){
            j[] <- j+1
            B[j] <- matVt[2,1]
            names(B)[j] <- "trend"
            if(Ttype=="A"){
                Bl[j] <- -Inf
                Bu[j] <- Inf
            }
            else{
                Bl[j] <- 0
                Bu[j] <- 2
            }
        }
        if(modelIsSeasonal && any(initialSeasonalEstimate)){
            if(componentsNumberETSSeasonal>1){
                for(k in 1:componentsNumberETSSeasonal){
                    if(initialSeasonalEstimate[k]){
                        B[j+2:lagsModel[componentsNumberETSNonSeasonal+k]-1] <-
                            matVt[componentsNumberETSNonSeasonal+k,
                                  2:lagsModel[componentsNumberETSNonSeasonal+k]-1]
                        names(B)[j+2:(lagsModel[componentsNumberETSNonSeasonal+k])-1] <-
                            paste0("seasonal",k,"_",
                                   2:lagsModel[componentsNumberETSNonSeasonal+k]-1)
                        if(Stype=="A"){
                            Bl[j+2:lagsModel[componentsNumberETSNonSeasonal+k]-1] <- -Inf
                            Bu[j+2:lagsModel[componentsNumberETSNonSeasonal+k]-1] <- Inf
                        }
                        else{
                            Bl[j+2:lagsModel[componentsNumberETSNonSeasonal+k]-1] <- 0
                            Bu[j+2:lagsModel[componentsNumberETSNonSeasonal+k]-1] <- Inf
                        }
                        j[] <- j+(lagsModelSeasonal[k]-1)
                    }
                }
            }
            else{
                B[j+2:(lagsModel[componentsNumberETS])-1] <-
                    matVt[componentsNumberETS,2:lagsModel[componentsNumberETS]-1]
                names(B)[j+2:(lagsModel[componentsNumberETS])-1] <-
                    paste0("seasonal_",2:lagsModel[componentsNumberETS]-1)
                if(Stype=="A"){
                    Bl[j+2:(lagsModel[componentsNumberETS])-1] <- -Inf
                    Bu[j+2:(lagsModel[componentsNumberETS])-1] <- Inf
                }
                else{
                    Bl[j+2:(lagsModel[componentsNumberETS])-1] <- 0
                    Bu[j+2:(lagsModel[componentsNumberETS])-1] <- Inf
                }
                j[] <- j+(lagsModel[componentsNumberETS]-1)
            }
        }
    }

    # ARIMA initials
    if(arimaModel && all(initialType!=c("complete","backcasting")) && initialArimaEstimate){
        B[j+1:initialArimaNumber] <-
            head(matVt[componentsNumberETS+componentsNumberARIMA,1:lagsModelMax],
                 initialArimaNumber)
        names(B)[j+1:initialArimaNumber] <- paste0("ARIMAState",1:initialArimaNumber)

        # Fix initial state if the polynomial is not zero
        if(tail(arimaPolynomials$ariPolynomial,1)!=0){
            B[j+1:initialArimaNumber] <- B[j+1:initialArimaNumber] /
                tail(arimaPolynomials$ariPolynomial,1)
        }

        if(Etype=="A"){
            Bl[j+1:initialArimaNumber] <- -Inf
            Bu[j+1:initialArimaNumber] <- Inf
        }
        else{
            # Make sure that ARIMA states are positive to avoid errors
            B[j+1:initialArimaNumber] <- abs(B[j+1:initialArimaNumber])
            Bl[j+1:initialArimaNumber] <- 0
            Bu[j+1:initialArimaNumber] <- Inf
        }
        j[] <- j+initialArimaNumber
    }

    # Initials of the xreg
    if(initialType!="complete" && initialXregEstimate){
        xregNumberToEstimate <- sum(xregParametersEstimated)
        B[j+1:xregNumberToEstimate] <-
            matVt[componentsNumberETS+componentsNumberARIMA+
                      which(xregParametersEstimated==1),1]
        names(B)[j+1:xregNumberToEstimate] <-
            rownames(matVt)[componentsNumberETS+componentsNumberARIMA+
                                which(xregParametersEstimated==1)]
        if(Etype=="A"){
            Bl[j+1:xregNumberToEstimate] <- -Inf
            Bu[j+1:xregNumberToEstimate] <- Inf
        }
        else{
            Bl[j+1:xregNumberToEstimate] <- -Inf
            Bu[j+1:xregNumberToEstimate] <- Inf
        }
        j[] <- j+xregNumberToEstimate
    }

    if(constantEstimate){
        j[] <- j+1
        B[j] <- matVt[componentsNumberETS+componentsNumberARIMA+xregNumber+1,1]
        names(B)[j] <- constantName
        if(etsModel || sum(iOrders)!=0){
            if(Etype=="A"){
                Bu[j] <- quantile(diff(yInSample[otLogical]),0.6)
                Bl[j] <- -Bu[j]
            }
            else{
                Bu[j] <- exp(quantile(diff(log(yInSample[otLogical])),0.6))
                Bl[j] <- exp(quantile(diff(log(yInSample[otLogical])),0.4))
            }

            # Failsafe for weird cases, when upper bound is the same or lower than the lower one
            if(Bu[j]<=Bl[j]){
                Bu[j] <- Inf
                Bl[j] <- switch(Etype,"A"=-Inf,"M"=0)
            }

            # Failsafe for cases, when the B is outside of bounds
            if(B[j]<=Bl[j]){
                Bl[j] <- switch(Etype,"A"=-Inf,"M"=0)
            }
            if(B[j]>=Bu[j]){
                Bu[j] <- Inf
            }
        }
        else{
            Bu[j] <- max(abs(yInSample[otLogical]),abs(B[j])*1.01)
            Bl[j] <- -Bu[j]
        }
    }

    # Add lambda if it is needed
    if(otherParameterEstimate){
        j[] <- j+1
        B[j] <- other
        names(B)[j] <- "other"
        Bl[j] <- 1e-10
        Bu[j] <- Inf
    }

    return(list(B=B,Bl=Bl,Bu=Bu))
}

#' @keywords internal
adam_scaler <- function(distribution, Etype, errors, yFitted, obsInSample, other){
    return(switch(distribution,
                  "dnorm"=sqrt(sum(errors^2)/obsInSample),
                  "dlaplace"=sum(abs(errors))/obsInSample,
                  "ds"=sum(sqrt(abs(errors))) / (obsInSample*2),
                  "dgnorm"=(other*sum(abs(errors)^other)/obsInSample)^{1/other},
                  "dalaplace"=sum(errors*(other-(errors<=0)*1))/obsInSample,
                  "dlnorm"=sqrt(2*abs(switch(Etype,
                                             "A"=1-sqrt(abs(1-sum(log(abs(1+errors/yFitted))^2)/
                                                             obsInSample)),
                                             "M"=1-sqrt(abs(1-sum(log(1+errors)^2)/obsInSample))))),
                  "dllaplace"=switch(Etype,
                                     "A"=Re(sum(abs(log(as.complex(
                                         1+errors/yFitted))))/obsInSample),
                                     "M"=sum(abs(log(1+errors))/obsInSample)),
                  "dls"=switch(Etype,
                               "A"=Re(sum(sqrt(abs(log(as.complex(
                                   1+errors/yFitted))))/obsInSample)),
                               "M"=sum(sqrt(abs(log(1+errors)))/obsInSample)),
                  "dlgnorm"=switch(Etype,
                                   "A"=Re((other*sum(abs(log(as.complex(1+errors/yFitted)))^other)/
                                               obsInSample)^{1/other}),
                                   "M"=(other*sum(abs(log(as.complex(1+errors)))^other)/
                                            obsInSample)^{1/other}),
                  "dinvgauss"=switch(Etype,
                                     "A"=sum((errors/yFitted)^2/(1+errors/yFitted))/obsInSample,
                                     "M"=sum((errors)^2/(1+errors))/obsInSample),
                  "dgamma"=switch(Etype,
                                  "A"=sum((errors/yFitted)^2)/obsInSample,
                                  "M"=sum(errors^2)/obsInSample)))
}

#### IC weights (Akaike weights) ####
#' @keywords internal
adam_ic_weights <- function(icSelection, threshold=1e-5){
    icBest <- min(icSelection);
    weights <- exp(-0.5*(icSelection - icBest)) / sum(exp(-0.5*(icSelection - icBest)));
    weights[weights < threshold] <- 0;
    weights <- weights / sum(weights);
    return(weights);
}

#### Bounds checker for ARIMA stationarity and ETS smoothing parameter constraints ####
#' @keywords internal
adam_bounds_checker <- function(adamElements, arimaPolynomials,
                                bounds,
                                etsModel, modelIsTrendy, modelIsSeasonal,
                                componentsNumberETS, componentsNumberETSNonSeasonal,
                                componentsNumberETSSeasonal,
                                arimaModel, arEstimate, maEstimate,
                                xregModel, regressors, xregNumber, componentsNumberARIMA,
                                lagsModelAll, obsInSample,
                                arPolynomialMatrix, maPolynomialMatrix,
                                phiEstimate){
    if(bounds=="usual"){
        if(arimaModel && any(c(arEstimate, maEstimate))){
            if(arEstimate &&
               (all(-arimaPolynomials$arPolynomial[-1]>0) &
                sum(-(arimaPolynomials$arPolynomial[-1]))>=1)){
                arPolynomialMatrix[,1] <- -arimaPolynomials$arPolynomial[-1];
                arPolyroots <- abs(eigen(arPolynomialMatrix, symmetric=FALSE,
                                        only.values=TRUE)$values);
                if(any(arPolyroots>1)){
                    return(1E+100*max(arPolyroots));
                }
            }
            if(maEstimate && sum(arimaPolynomials$maPolynomial[-1])>=1){
                maPolynomialMatrix[,1] <- arimaPolynomials$maPolynomial[-1];
                maPolyroots <- abs(eigen(maPolynomialMatrix, symmetric=FALSE,
                                        only.values=TRUE)$values);
                if(any(maPolyroots>1)){
                    return(1E+100*max(abs(maPolyroots)));
                }
            }
        }

        if(etsModel){
            if(any(adamElements$vecG[1:componentsNumberETS]>1) ||
               any(adamElements$vecG[1:componentsNumberETS]<0)){
                return(1E+300);
            }
            if(modelIsTrendy){
                if(adamElements$vecG[2] > adamElements$vecG[1]){
                    return(1E+300);
                }
                if(modelIsSeasonal &&
                   any(adamElements$vecG[componentsNumberETSNonSeasonal+c(1:componentsNumberETSSeasonal)] >
                       (1-adamElements$vecG[1]))){
                    return(1E+300);
                }
            }
            else{
                if(modelIsSeasonal &&
                   any(adamElements$vecG[componentsNumberETSNonSeasonal+c(1:componentsNumberETSSeasonal)] >
                       (1-adamElements$vecG[1]))){
                    return(1E+300);
                }
            }
            if(phiEstimate && (adamElements$matF[2,2]>1 || adamElements$matF[2,2]<0)){
                return(1E+300);
            }
        }

        if(xregModel && regressors=="adapt"){
            if(any(adamElements$vecG[componentsNumberETS+componentsNumberARIMA+1:xregNumber]>1) ||
               any(adamElements$vecG[componentsNumberETS+componentsNumberARIMA+1:xregNumber]<0)){
                return(1E+100*max(abs(
                    adamElements$vecG[componentsNumberETS+componentsNumberARIMA+1:xregNumber]-0.5
                )));
            }
        }
    }
    else if(bounds=="admissible"){
        if(arimaModel){
            if(arEstimate &&
               (all(-arimaPolynomials$arPolynomial[-1]>0) &
                sum(-(arimaPolynomials$arPolynomial[-1]))>=1)){
                arPolynomialMatrix[,1] <- -arimaPolynomials$arPolynomial[-1];
                eigenValues <- abs(eigen(arPolynomialMatrix, symmetric=FALSE,
                                        only.values=TRUE)$values);
                if(any(eigenValues>1)){
                    return(1E+100*max(eigenValues));
                }
            }
        }
        eigenValues <- smoothEigens(adamElements$vecG, adamElements$matF,
                                    adamElements$matWt, lagsModelAll, xregModel, obsInSample);
        if(any(eigenValues>1+1E-50)){
            return(1E+100*max(eigenValues));
        }
    }
    return(0);
}

#### xreg variable selector using stepwise regression on residuals ####
#' @keywords internal
adam_xreg_selector <- function(errors, xregData, obsInSample, ic, df, distribution,
                                occurrence, other){
    alpha <- shape <- nu <- NULL;
    if(distribution=="dalaplace"){
        alpha <- other;
    }
    else if(any(distribution==c("dgnorm","dlgnorm"))){
        shape <- other;
    }
    else if(distribution=="dt"){
        nu <- other;
    }
    stepwiseModel <- suppressWarnings(
        stepwise(data.frame(errorsIvan41=errors, xregData[1:obsInSample,,drop=FALSE]),
                 ic=ic, df=df, distribution=distribution, occurrence=occurrence, silent=TRUE,
                 alpha=alpha, shape=shape, nu=nu)
    );
    return(list(initialXreg=coef(stepwiseModel)[-1], other=stepwiseModel$other,
                formula=formula(stepwiseModel)));
}

#### Model name assembler ####
#' @keywords internal
adam_model_name <- function(etsModel, model, xregModel, arimaModel,
                             arOrders, iOrders, maOrders, lags,
                             regressors, constantRequired, constantName,
                             occurrence, componentsNumberETSSeasonal){
    modelName <- "";
    if(etsModel){
        if(model!="NNN"){
            modelName <- "ETS";
            if(xregModel){
                modelName <- paste0(modelName,"X");
            }
            modelName <- paste0(modelName,"(",model,")");
            if(componentsNumberETSSeasonal>1){
                modelName <- paste0(modelName,"[",
                                    paste0(lags[lags!=1], collapse=", "),"]");
            }
        }
    }
    if(arimaModel){
        if(etsModel){
            modelName <- paste0(modelName,"+");
        }
        if(all(lags==1) || (all(arOrders[lags>1]==0) && all(iOrders[lags>1]==0) &&
                            all(maOrders[lags>1]==0))){
            modelName <- paste0(modelName,"ARIMA");
            if(!etsModel && xregModel){
                modelName <- paste0(modelName,"X");
            }
            modelName <- paste0(modelName,"(",arOrders[1],",",iOrders[1],",",
                                maOrders[1],")");
        }
        else{
            modelName <- paste0(modelName,"SARIMA");
            if(!etsModel && xregModel){
                modelName <- paste0(modelName,"X");
            }
            for(i in 1:length(arOrders)){
                if(all(arOrders[i]==0) && all(iOrders[i]==0) && all(maOrders[i]==0)){
                    next;
                }
                modelName <- paste0(modelName,"(",arOrders[i],",",iOrders[i],",",
                                    maOrders[i],")[",lags[i],"]");
            }
        }
    }
    if(regressors=="adapt"){
        modelName <- paste0(modelName,"{D}");
    }
    if(!etsModel && !arimaModel){
        if(model=="NNN"){
            modelName <- "Constant level";
        }
        else if(regressors=="adapt"){
            modelName <- "Dynamic regression";
        }
        else{
            modelName <- "Regression";
        }
    }
    else{
        if(constantRequired){
            modelName <- paste0(modelName," with ",constantName);
        }
    }
    if(all(occurrence!=c("n","none"))){
        modelName <- paste0("i",modelName,
                            switch(occurrence,
                                   "f"=,"fixed"="[F]",
                                   "d"=,"direct"="[D]",
                                   "o"=,"odds-ratio"="[O]",
                                   "i"=,"inverse-odds-ratio"="[I]",
                                   "g"=,"general"="[G]",
                                   ""));
    }
    return(modelName);
}

#### Initial values collector ####
#' @keywords internal
adam_initial_collector <- function(matVt, etsModel, modelIsTrendy, modelIsSeasonal,
                                  lagsModel, lagsModelMax,
                                  initialLevelEstimate, initialTrendEstimate,
                                  initialSeasonalEstimate,
                                  componentsNumberETSSeasonal,
                                  arimaModel, initialArimaEstimate, initialArima,
                                  initialArimaNumber,
                                  componentsNumberETS, componentsNumberARIMA,
                                  arimaPolynomials, Etype,
                                  xregModel, initialXregEstimate, xregNumber){
    initialValue <- vector("list", etsModel*(1+modelIsTrendy+modelIsSeasonal)+arimaModel+xregModel);
    initialValueETS <- vector("list", etsModel*length(lagsModel));
    initialValueNames <- vector("character", etsModel*(1+modelIsTrendy+modelIsSeasonal)+arimaModel+xregModel);
    # The vector that defines what was estimated in the model
    initialEstimated <- vector("logical", etsModel*(1+modelIsTrendy+modelIsSeasonal*componentsNumberETSSeasonal)+
                                   arimaModel+xregModel);

    # Write down the initials of ETS
    j <- 0;
    if(etsModel){
        # Write down level, trend and seasonal
        for(i in 1:length(lagsModel)){
            # In case of level / trend, we want to get the very first value
            if(lagsModel[i]==1){
                initialValueETS[[i]] <- head(matVt[i,1:lagsModelMax],1);
            }
            # In cases of seasonal components, they should be at the end of the pre-heat period
            else{
                initialValueETS[[i]] <- tail(matVt[i,1:lagsModelMax],lagsModel[i]);
            }
        }
        j[] <- j+1;
        # Write down level in the final list
        initialEstimated[j] <- initialLevelEstimate;
        initialValue[[j]] <- initialValueETS[[j]];
        initialValueNames[j] <- c("level");
        names(initialEstimated)[j] <- initialValueNames[j];
        if(modelIsTrendy){
            j[] <- 2;
            initialEstimated[j] <- initialTrendEstimate;
            # Write down trend in the final list
            initialValue[[j]] <- initialValueETS[[j]];
            # Remove the trend from ETS list
            initialValueETS[[j]] <- NULL;
            initialValueNames[j] <- c("trend");
            names(initialEstimated)[j] <- initialValueNames[j];
        }
        # Write down the initial seasonals
        if(modelIsSeasonal){
            initialEstimated[j+c(1:componentsNumberETSSeasonal)] <- initialSeasonalEstimate;
            # Remove the level from ETS list
            initialValueETS[[1]] <- NULL;
            j[] <- j+1;
            if(length(initialSeasonalEstimate)>1){
                initialValue[[j]] <- initialValueETS;
                initialValueNames[[j]] <- "seasonal";
                names(initialEstimated)[j+0:(componentsNumberETSSeasonal-1)] <-
                    paste0(initialValueNames[j],c(1:componentsNumberETSSeasonal));
            }
            else{
                initialValue[[j]] <- initialValueETS[[1]];
                initialValueNames[[j]] <- "seasonal";
                names(initialEstimated)[j] <- initialValueNames[j];
            }
        }
    }

    # Write down the ARIMA initials
    if(arimaModel){
        j[] <- j+1;
        initialEstimated[j] <- initialArimaEstimate;
        if(initialArimaEstimate){
            initialValue[[j]] <- head(matVt[componentsNumberETS+componentsNumberARIMA,],initialArimaNumber);
            # Fix the values to get proper initials, not just the values of states
            if(tail(arimaPolynomials$ariPolynomial,1)!=0){
                initialValue[[j]] <- switch(Etype,
                                            "A"=initialValue[[j]] / tail(arimaPolynomials$ariPolynomial,1),
                                            "M"=exp(log(initialValue[[j]]) / tail(arimaPolynomials$ariPolynomial,1)));
            }
        }
        else{
            initialValue[[j]] <- initialArima;
        }
        initialValueNames[j] <- "arima";
        names(initialEstimated)[j] <- initialValueNames[j];
    }
    # Write down the xreg initials
    if(xregModel){
        j[] <- j+1;
        initialEstimated[j] <- initialXregEstimate;
        initialValue[[j]] <- matVt[componentsNumberETS+componentsNumberARIMA+1:xregNumber,lagsModelMax];
        initialValueNames[j] <- "xreg";
        names(initialEstimated)[j] <- initialValueNames[j];
    }
    names(initialValue) <- initialValueNames;

    return(list(initialValue=initialValue, initialEstimated=initialEstimated));
}

#### ETS model selector (branch-and-bound + full pool) ####
#' @keywords internal
adam_selector <- function(estimator_fn, model, modelsPool, allowMultiplicative,
                           etsModel, Etype, Ttype, Stype, damped, lags,
                           lagsModelSeasonal, lagsModelARIMA,
                           obsStates, obsInSample,
                           yInSample, persistence, persistenceEstimate,
                           persistenceLevel, persistenceLevelEstimate,
                           persistenceTrend, persistenceTrendEstimate,
                           persistenceSeasonal, persistenceSeasonalEstimate,
                           persistenceXreg, persistenceXregEstimate, persistenceXregProvided,
                           phi, phiEstimate,
                           initialType, initialLevel, initialTrend, initialSeasonal,
                           initialArima, initialEstimate,
                           initialLevelEstimate, initialTrendEstimate, initialSeasonalEstimate,
                           initialArimaEstimate, initialXregEstimate, initialXregProvided,
                           arimaModel, arRequired, iRequired, maRequired, armaParameters,
                           componentsNumberARIMA, componentsNamesARIMA,
                           formula, xregModel, xregModelInitials, xregData,
                           xregNumber, xregNames, regressors,
                           xregParametersMissing, xregParametersIncluded,
                           xregParametersEstimated, xregParametersPersistence,
                           constantRequired, constantEstimate, constantValue, constantName,
                           ot, otLogical, occurrenceModel, pFitted, icFunction,
                           bounds, loss, lossFunction, distribution,
                           horizon, multisteps, other, otherParameterEstimate, lambda,
                           silent, B){
    if(is.null(modelsPool)){
        if(!silent){
            cat("Forming the pool of models based on... ");
        }

        if(!allowMultiplicative){
            poolErrors <- c("A");
            poolTrends <- c("N","A","Ad");
            poolSeasonals <- c("N","A");
        }
        else{
            poolErrors <- c("A","M");
            poolTrends <- c("N","A","Ad","M","Md");
            poolSeasonals <- c("N","A","M");
        }

        if(Etype!="Z"){
            poolErrors <- poolErrorsSmall <- Etype;
        }
        else{
            poolErrorsSmall <- "A";
        }

        if(Ttype!="Z"){
            if(Ttype=="X"){
                poolTrendsSmall <- c("N","A");
                poolTrends <- c("N","A","Ad");
                checkTrend <- TRUE;
            }
            else if(Ttype=="Y"){
                poolTrendsSmall <- c("N","M");
                poolTrends <- c("N","M","Md");
                checkTrend <- TRUE;
            }
            else{
                if(damped){
                    poolTrends <- poolTrendsSmall <- paste0(Ttype,"d");
                }
                else{
                    poolTrends <- poolTrendsSmall <- Ttype;
                }
                checkTrend <- FALSE;
            }
        }
        else{
            poolTrendsSmall <- c("N","A");
            checkTrend <- TRUE;
        }

        if(Stype!="Z"){
            if(Stype=="X"){
                poolSeasonals <- poolSeasonalsSmall <- c("N","A");
                checkSeasonal <- TRUE;
            }
            else if(Stype=="Y"){
                poolSeasonalsSmall <- c("N","M");
                poolSeasonals <- c("N","M");
                checkSeasonal <- TRUE;
            }
            else{
                poolSeasonalsSmall <- Stype;
                poolSeasonals <- Stype;
                checkSeasonal <- FALSE;
            }
        }
        else{
            poolSeasonalsSmall <- c("N","A","M");
            checkSeasonal <- TRUE;
        }

        poolSmall <- paste0(rep(poolErrorsSmall, length(poolTrendsSmall)*length(poolSeasonalsSmall)),
                            rep(poolTrendsSmall, each=length(poolSeasonalsSmall)),
                            rep(poolSeasonalsSmall, length(poolTrendsSmall)));
        if(any(substr(poolSmall,3,3)=="M") && all(Etype!=c("A","X"))){
            multiplicativeSeason <- (substr(poolSmall,3,3)=="M");
            poolSmall[multiplicativeSeason] <- paste0("M", substr(poolSmall[multiplicativeSeason],2,3));
        }
        modelsTested <- NULL;
        modelCurrent <- NA;

        j <- 1;
        i <- 0;
        check <- TRUE;
        besti <- bestj <- 1;
        results <- vector("list", length(poolSmall));

        while(check){
            i <- i + 1;
            modelCurrent[] <- poolSmall[j];
            if(!silent){
                cat(modelCurrent,"\b, ");
            }
            Etype[] <- substring(modelCurrent,1,1);
            Ttype[] <- substring(modelCurrent,2,2);
            if(nchar(modelCurrent)==4){
                phi[] <- 0.95;
                phiEstimate[] <- TRUE;
                Stype[] <- substring(modelCurrent,4,4);
            }
            else{
                phi[] <- 1;
                phiEstimate[] <- FALSE;
                Stype[] <- substring(modelCurrent,3,3);
            }

            results[[i]] <- estimator_fn(etsModel, Etype, Ttype, Stype, lags,
                                         lagsModelSeasonal, lagsModelARIMA,
                                         obsStates, obsInSample,
                                         yInSample, persistence, persistenceEstimate,
                                         persistenceLevel, persistenceLevelEstimate,
                                         persistenceTrend, persistenceTrendEstimate,
                                         persistenceSeasonal, persistenceSeasonalEstimate,
                                         persistenceXreg, persistenceXregEstimate,
                                         persistenceXregProvided,
                                         phi, phiEstimate,
                                         initialType, initialLevel, initialTrend, initialSeasonal,
                                         initialArima, initialEstimate,
                                         initialLevelEstimate, initialTrendEstimate,
                                         initialSeasonalEstimate,
                                         initialArimaEstimate, initialXregEstimate,
                                         initialXregProvided,
                                         arimaModel, arRequired, iRequired, maRequired,
                                         armaParameters,
                                         componentsNumberARIMA, componentsNamesARIMA,
                                         formula, xregModel, xregModelInitials, xregData,
                                         xregNumber, xregNames, regressors,
                                         xregParametersMissing, xregParametersIncluded,
                                         xregParametersEstimated, xregParametersPersistence,
                                         constantRequired, constantEstimate, constantValue,
                                         constantName,
                                         ot, otLogical, occurrenceModel, pFitted,
                                         bounds, loss, lossFunction, distribution,
                                         horizon, multisteps, other, otherParameterEstimate,
                                         lambda, B);
            results[[i]]$IC <- icFunction(results[[i]]$logLikADAMValue);
            results[[i]]$Etype <- Etype;
            results[[i]]$Ttype <- Ttype;
            results[[i]]$Stype <- Stype;
            results[[i]]$phiEstimate <- phiEstimate;
            if(phiEstimate){
                results[[i]]$phi <- results[[i]]$B[names(results[[i]]$B)=="phi"];
            }
            else{
                results[[i]]$phi <- 1;
            }
            results[[i]]$model <- modelCurrent;
            modelsTested <- c(modelsTested, modelCurrent);

            if(j>1){
                if(results[[besti]]$IC <= results[[i]]$IC){
                    if(substring(modelCurrent,2,2)==substring(poolSmall[bestj],2,2)){
                        poolSeasonals <- results[[besti]]$Stype;
                        checkSeasonal <- FALSE;
                        j <- which(poolSmall!=poolSmall[bestj] &
                                       substring(poolSmall,nchar(poolSmall),nchar(poolSmall))==poolSeasonals);
                    }
                    else{
                        poolTrends <- results[[bestj]]$Ttype;
                        checkTrend[] <- FALSE;
                    }
                }
                else{
                    if(substring(modelCurrent,2,2) == substring(poolSmall[besti],2,2)){
                        poolSeasonals <- poolSeasonals[poolSeasonals!=results[[besti]]$Stype];
                        if(length(poolSeasonals)>1){
                            bestj[] <- j;
                            besti[] <- i;
                            j <- 3;
                        }
                        else{
                            bestj[] <- j;
                            besti[] <- i;
                            j <- which(substring(poolSmall,nchar(poolSmall),nchar(poolSmall))==poolSeasonals &
                                           substring(poolSmall,2,2)!=substring(modelCurrent,2,2));
                            checkSeasonal[] <- FALSE;
                        }
                    }
                    else{
                        poolTrends <- poolTrends[poolTrends!=results[[bestj]]$Ttype];
                        besti[] <- i;
                        bestj[] <- j;
                        checkTrend[] <- FALSE;
                    }
                }

                if(all(!c(checkTrend,checkSeasonal))){
                    check[] <- FALSE;
                }
            }
            else{
                j <- 2;
            }

            if(length(j)==0){
                j <- length(poolSmall);
            }
            if(j>length(poolSmall)){
                check[] <- FALSE;
            }
        }

        modelsPool <- unique(c(modelsTested,
                               paste0(rep(poolErrors, each=length(poolTrends)*length(poolSeasonals)),
                                      poolTrends,
                                      rep(poolSeasonals, each=length(poolTrends)))));
        j <- length(modelsTested);
    }
    else{
        j <- 0;
        results <- vector("list", length(modelsPool));
    }
    modelsNumber <- length(modelsPool);

    if(!silent){
        cat("Estimation progress:    ");
    }
    while(j < modelsNumber){
        j <- j + 1;
        if(!silent){
            if(j==1){
                cat("\b");
            }
            cat(paste0(rep("\b",nchar(round((j-1)/modelsNumber,2)*100)+1),collapse=""));
            cat(round(j/modelsNumber,2)*100,"\b%");
        }

        modelCurrent <- modelsPool[j];
        Etype <- substring(modelCurrent,1,1);
        Ttype <- substring(modelCurrent,2,2);
        if(nchar(modelCurrent)==4){
            phi[] <- 0.95;
            Stype <- substring(modelCurrent,4,4);
            phiEstimate <- TRUE;
        }
        else{
            phi[] <- 1;
            Stype <- substring(modelCurrent,3,3);
            phiEstimate <- FALSE;
        }

        results[[j]] <- estimator_fn(etsModel, Etype, Ttype, Stype, lags,
                                     lagsModelSeasonal, lagsModelARIMA,
                                     obsStates, obsInSample,
                                     yInSample, persistence, persistenceEstimate,
                                     persistenceLevel, persistenceLevelEstimate,
                                     persistenceTrend, persistenceTrendEstimate,
                                     persistenceSeasonal, persistenceSeasonalEstimate,
                                     persistenceXreg, persistenceXregEstimate,
                                     persistenceXregProvided,
                                     phi, phiEstimate,
                                     initialType, initialLevel, initialTrend, initialSeasonal,
                                     initialArima, initialEstimate,
                                     initialLevelEstimate, initialTrendEstimate,
                                     initialSeasonalEstimate,
                                     initialArimaEstimate, initialXregEstimate,
                                     initialXregProvided,
                                     arimaModel, arRequired, iRequired, maRequired,
                                     armaParameters,
                                     componentsNumberARIMA, componentsNamesARIMA,
                                     formula, xregModel, xregModelInitials, xregData,
                                     xregNumber, xregNames, regressors,
                                     xregParametersMissing, xregParametersIncluded,
                                     xregParametersEstimated, xregParametersPersistence,
                                     constantRequired, constantEstimate, constantValue,
                                     constantName,
                                     ot, otLogical, occurrenceModel, pFitted,
                                     bounds, loss, lossFunction, distribution,
                                     horizon, multisteps, other, otherParameterEstimate,
                                     lambda, B);
        results[[j]]$IC <- icFunction(results[[j]]$logLikADAMValue);
        results[[j]]$Etype <- Etype;
        results[[j]]$Ttype <- Ttype;
        results[[j]]$Stype <- Stype;
        results[[j]]$phiEstimate <- phiEstimate;
        if(phiEstimate){
            results[[j]]$phi <- results[[j]]$B[names(results[[j]]$B)=="phi"];
        }
        else{
            results[[j]]$phi <- 1;
        }
        results[[j]]$model <- modelCurrent;
    }

    if(!silent){
        cat("... Done! \n");
    }

    icSelection <- vector("numeric", modelsNumber);
    for(i in 1:modelsNumber){
        icSelection[i] <- results[[i]]$IC;
    }
    names(icSelection) <- modelsPool;
    icSelection[is.nan(icSelection)] <- 1E100;

    return(list(results=results, icSelection=icSelection));
}
