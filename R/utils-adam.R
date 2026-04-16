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
