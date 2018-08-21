#' Functions that extract values from the fitted model
#'
#' These functions allow extracting orders and lags for \code{ssarima()}, \code{gum()} and \code{sma()},
#' type of model from \code{es()} and \code{ces()} and name of model.
#'
#' \code{orders()} and \code{lags()} are useful only for SSARIMA, GUM and SMA. They return \code{NA} for other functions.
#' This can also be applied to \code{arima()}, \code{Arima()} and \code{auto.arima()} functions from stats and forecast packages.
#' \code{modelType()} is useful only for ETS and CES. They return \code{NA} for other functions.
#' This can also be applied to \code{ets()} function from forecast package. \code{errorType}
#' extracts the type of error from the model (either additive or multiplicative). Finally, \code{modelName}
#' returns the name of the fitted model. For example, "ARIMA(0,1,1)". This is purely descriptive and
#' can also be applied to non-smooth classes, so that, for example, you can easily extract the name
#' of the fitted AR model from \code{ar()} function from \code{stats} package.
#'
#' @template ssAuthor
#' @template ssKeywords
#'
#' @aliases orders
#' @param object Model estimated using one of the functions of smooth package.
#' @param ... Currently nothing is accepted via ellipsis.
#' @return     Either vector, scalar or list with values is returned.
#' \code{orders()} in case of ssarima returns list of values:
#' \itemize{
#' \item \code{ar} - AR orders.
#' \item \code{i} - I orders.
#' \item \code{ma} - MA orders.
#' }
#' \code{lags()} returns the vector of lags of the model.
#' All the other functions return strings of character.
#' @seealso \link[forecast]{forecast}, \link[smooth]{ssarima}
#' @examples
#'
#' x <- rnorm(100,0,1)
#'
#' # Just as example. orders and lags do not return anything for ces() and es(). But modelType() does.
#' ourModel <- ces(x, h=10)
#' orders(ourModel)
#' lags(ourModel)
#' modelType(ourModel)
#' modelName(ourModel)
#'
#' # And as another example it does the opposite for gum() and ssarima()
#' ourModel <- gum(x, h=10, orders=c(1,1), lags=c(1,4))
#' orders(ourModel)
#' lags(ourModel)
#' modelType(ourModel)
#' modelName(ourModel)
#'
#' # Finally these values can be used for simulate functions or original functions.
#' ourModel <- auto.ssarima(x)
#' ssarima(x, orders=orders(ourModel), lags=lags(ourModel), constant=ourModel$constant)
#' sim.ssarima(orders=orders(ourModel), lags=lags(ourModel), constant=ourModel$constant)
#'
#' ourModel <- es(x)
#' es(x, model=modelType(ourModel))
#' sim.es(model=modelType(ourModel))
#'
#' @rdname orders
#' @export orders
orders <- function(object, ...) UseMethod("orders")

#' @aliases lags
#' @rdname orders
#' @export lags
lags <- function(object, ...) UseMethod("lags")

#' @aliases modelName
#' @rdname orders
#' @export modelName
modelName <- function(object, ...) UseMethod("modelName")

#' @aliases modelType
#' @rdname orders
#' @export modelType
modelType <- function(object, ...) UseMethod("modelType")

modelLags <- function(object, ...) UseMethod("modelLags")

smoothType <- function(object, ...) UseMethod("smoothType")

##### Likelihood function and stuff #####

#' @importFrom greybox AICc
#' @export
AICc.smooth <- function(object, ...){
    llikelihood <- logLik(object);
    nParamAll <- nParam(object);
    llikelihood <- llikelihood[1:length(llikelihood)];

    if(!is.null(object$imodel)){
        obs <- sum(object$fitted!=0);
        nParamSizes <- nParamAll - object$nParam[1,3];
        IC <- (2*nParamAll - 2*llikelihood +
                   2*nParamSizes*(nParamSizes + 1) / (obs - nParamSizes - 1));
    }
    else{
        obs <- nobs(object);
        IC <- 2*nParamAll - 2*llikelihood + 2 * nParamAll * (nParamAll + 1) / (obs - nParamAll - 1);
    }

    return(IC);
}

#' @importFrom greybox BICc
#' @export
BICc.smooth <- function(object, ...){
    llikelihood <- logLik(object);
    nParamAll <- nParam(object);
    llikelihood <- llikelihood[1:length(llikelihood)];

    if(!is.null(object$imodel)){
        obs <- sum(object$fitted!=0);
        nParamSizes <- nParamAll - object$nParam[1,3];
        IC <- - 2*llikelihood + (nParamSizes * log(obs) * obs) / (obs - nParamSizes - 1);
    }
    else{
        obs <- nobs(object);
        IC <- - 2*llikelihood + (nParamAll * log(obs) * obs) / (obs - nParamAll - 1);
    }

    return(IC);
}

#' Function returns the covariance matrix of conditional multiple steps ahead forecast errors
#'
#' This function extracts covariance matrix of 1 to h steps ahead forecast errors for
#' \code{ssarima()}, \code{gum()}, \code{sma()}, \code{es()} and \code{ces()} models.
#'
#' The function returns either scalar (if it is a non-smooth model)
#' or the matrix of (h x h) size with variances and covariances of 1 to h steps ahead
#' forecast errors. This is currently done based on empirical values. The analytical ones
#' are more complicated.
#'
#' @template ssAuthor
#' @template ssKeywords
#'
#' @param object Model estimated using one of the functions of smooth package.
#' @param type What method to use in order to produce covariance matrix:
#' \enumerate{
#' \item \code{analytical} - based on the state space structure of the model and the
#' one-step-ahead forecast error. This works for pure additive and pure multiplicative
#' models. The values for the mixed models might be off.
#' \item \code{empirical} - based on the in-sample 1 to h steps ahead forecast errors
#' (works fine on larger samples);
#' \item \code{simulated} - the data is simulated from the estimated model, then the
#' same model is applied to it and then the empirical 1 to h steps ahead forecast
#' errors are produced;
#' }
#' @param ... Other parameters passed to simulate function (if \code{type="simulated"}
#' is used). These are \code{obs}, \code{nsim} and \code{seed}. By default
#' \code{obs=1000}, \code{nsim=100}. This approach increases the accuracy of
#' covariance matrix on small samples and intermittent data;
#' @return Scalar in cases of non-smooth functions. (h x h) matrix otherwise.
#'
#' @seealso \link[smooth]{orders}
#' @examples
#'
#' x <- rnorm(100,0,1)
#'
#' # A simple example with a 5x5 covariance matrix
#' ourModel <- ces(x, h=5)
#' covar(ourModel)
#'
#' @rdname covar
#' @export covar
covar <-  function(object, type=c("analytical","empirical","simulated"), ...) UseMethod("covar")

#' @export
covar.default <- function(object, type=c("analytical","empirical","simulated"), ...){
    # Function extracts the conditional variances from the model
    return(sigma(object)^2);
}

#' @aliases covar.smooth
#' @rdname covar
#' @export
covar.smooth <- function(object, type=c("analytical","empirical","simulated"), ...){
    # Function extracts the conditional variances from the model

    if(any(class(object)=="smoothC")){
        stop("Sorry, but covariance matrix is not available for the combinations.",
            call.=FALSE)
    }

    type <- substr(type[1],1,1);

    if(is.null(object$persistence) & any(type==c("a","s"))){
        warning(paste0("The provided model does not contain the components necessary for the ",
                       "derivation of the covariance matrix.\n",
                       "Did you combine forecasts? Switching to 'empirical'"),
                call.=FALSE);
        type <- "e";
    }

    if(!is.null(object$imodel) & type=="e"){
        warning(paste0("Empirical covariance matrix can be very inaccurate in cases of ",
                       "intemittent models.\nWe recommend using type='s' or type='a' instead."),
                call.=FALSE);
    }

    # Empirical covariance matrix
    if(type=="e"){
        if(errorType(object)=="A"){
            errors <- object$errors;
        }
        else{
            errors <- log(1 + object$errors);
        }
        if(!is.null(object$imodel)){
            obs <- t((errors!=0)*1) %*% (errors!=0)*1;
            obs[obs==0] <- 1;
            df <- obs - nParam(object);
            df[df<=0] <- obs[df<=0];
        }
        else{
            obs <- matrix(nobs(object),ncol(errors),ncol(errors));
            df <- obs - nParam(object);
            df[df<=0] <- obs[df<=0];
        }
        covarMat <- t(errors) %*% errors / df;
    }
    # Simulated covariance matrix
    else if(type=="s"){
        smoothType <- smoothType(object);
        ellipsis <- list(...);
        if(any(names(ellipsis)=="obs")){
            obs <- ellipsis$obs;
        }
        else{
            obs <- length(getResponse(object));
        }
        if(any(names(ellipsis)=="nsim")){
            nsim <- ellipsis$nsim;
        }
        else{
            nsim <- 1000;
        }
        if(any(names(ellipsis)=="seed")){
            seed <- ellipsis$seed;
        }
        else{
            seed <- NULL;
        }

        h <- length(object$forecast);
        if(smoothType=="ETS"){
            smoothFunction <- es;
        }
        # GUM models
        else if(smoothType=="GUM"){
            smoothFunction <- gum;
        }
        # SSARIMA models
        else if(smoothType=="ARIMA"){
            smoothFunction <- ssarima;
        }
        # CES models
        else if(smoothType=="CES"){
            smoothFunction <- ces;
        }
        # SMA models
        else if(smoothType=="SMA"){
            smoothFunction <- sma;
        }

        covarArray <- array(NA,c(h,h,nsim));

        newData <- simulate(object, nsim=nsim, obs=obs, seed=seed);
        for(i in 1:nsim){
            # Apply the model to the simulated data
            smoothModel <- smoothFunction(newData$data[,i], model=object, h=h);
            # Remove first h-1 and last values.
            errors <- smoothModel$errors;

            # Transform errors if needed
            if(errorType(object)=="M"){
                # Remove zeroes if they are present
                errors <- errors[!apply(errors==0,1,any),];
                errors <- log(1 + errors);
            }

            # Calculate covariance matrix
            if(!is.null(object$imodel)){
                obsInSample <- t((errors!=0)*1) %*% (errors!=0)*1;
                obsInSample[obsInSample==0] <- 1;
            }
            else{
                obsInSample <- matrix(nrow(errors),ncol(errors),ncol(errors));
            }
            covarArray[,,i] <- t(errors) %*% errors / obsInSample;
        }

        covarMat <- apply(covarArray,c(1,2),mean);
    }
    # Analytical covariance matrix
    else if(type=="a"){
        if(!is.null(object$imodel)){
            ot <- (residuals(object)!=0)*1;
        }
        else{
            ot <- rep(1,length(residuals(object)));
        }
        h <- length(object$forecast);
        lagsModel <- modelLags(object);
        s2 <- sigma(object)^2;
        persistence <- matrix(object$persistence,length(object$persistence),1);
        transition <- object$transition;
        measurement <- object$measurement;

        covarMat <- covarAnal(lagsModel, h, measurement, transition, persistence, s2);

    }
    return(covarMat);
    # correlation matrix: covar(test) / sqrt(diag(covar(test)) %*% t(diag(covar(test))))
}

#' @importFrom stats logLik
#' @export
logLik.smooth <- function(object,...){
    if(is.null(object$logLik)){
        warning("The likelihood of this model is unavailable. Hint: did you use combinations?");
        return(NULL);
    }
    else{
        return(structure(object$logLik,nobs=nobs(object),df=nParam(object),class="logLik"));
    }
}
#' @export
logLik.smooth.sim <- function(object,...){
    obs <- nobs(object);
    return(structure(object$logLik,nobs=obs,df=0,class="logLik"));
}
#' @export
logLik.iss <- function(object,...){
    if(is.null(object$logLik)){
        warning("The likelihood of this model is unavailable.");
        return(NULL);
    }
    else{
        obs <- nobs(object);
        return(structure(object$logLik,nobs=obs,df=nParam(object),class="logLik"));
    }
}

#' @importFrom stats nobs
#' @export
nobs.smooth <- function(object, ...){
    return(length(object$fitted));
}
#' @method nobs smooth.sim
#' @export
nobs.smooth.sim <- function(object, ...){
    if(is.null(dim(object$data))){
        return(length(object$data));
    }
    else{
        return(nrow(object$data));
    }
}
#' @export
nobs.iss <- function(object, ...){
    return(length(object$fitted));
}

#' @importFrom greybox nParam

#' @export
nParam.smooth <- function(object, ...){
    if(is.null(object$nParam)){
        warning("Number of parameters of the model is unavailable. Hint: did you use combinations?",
                call.=FALSE);
        return(NULL);
    }
    else{
        return(object$nParam[1,4]);
    }
}

#' @export
nParam.iss <- function(object, ...){
    return(object$nParam);
}

#' Prediction Likelihood Score
#'
#' Function estimates Prediction Likelihood Score for the provided model
#'
#' Prediction likelihood score (PLS) is based on either normal or log-normal
#' distribution of errors. This is extracted from the provided model. The likelihood
#' based on the distribution of 1 to h steps ahead forecast errors is used in the process.
#'
#' @template ssAuthor
#' @template ssKeywords
#'
#' @param object The model estimated using smooth functions. This thing also accepts
#' other models (e.g. estimated using functions from forecast package), but may not always
#' work properly with them.
#' @param holdout The values for the holdout part of the sample. If the model was fitted
#' on the data with the \code{holdout=TRUE}, then the parameter is not needed.
#' @param ... Parameters passed to covar function. The function is called in order to get
#' the covariance matrix of 1 to h steps ahead forecast errors.
#'
#' @return A value of the log-likelihood.
#' @references \itemize{
#' %\item Eltoft, T., Taesu, K., Te-Won, L. (2006). On the multivariate Laplace
#' distribution. IEEE Signal Processing Letters. 13 (5): 300-303.
#' \doi{10.1109/LSP.2006.870353} - this is not yet used in the function.
#' \item Snyder, R. D., Ord, J. K., Beaumont, A., 2012. Forecasting the intermittent
#' demand for slow-moving inventories: A modelling approach. International
#' Journal of Forecasting 28 (2), 485-496.
#' \item Kolassa, S., 2016. Evaluating predictive count data distributions in retail
#' sales forecasting. International Journal of Forecasting 32 (3), 788-803..
#' }
#' @examples
#'
#' # Generate data, apply es() with the holdout parameter and calculate PLS
#' x <- rnorm(100,0,1)
#' ourModel <- es(x, h=10, holdout=TRUE, intervals=TRUE)
#' pls(ourModel, type="a")
#' pls(ourModel, type="e")
#' pls(ourModel, type="s", obs=100, nsim=100)
#'
#' @rdname pls
#' @export pls
pls <-  function(object, holdout=NULL, ...) UseMethod("pls")
# Function calculates PLS based on the provided model

#' @importFrom stats dnorm
#' @export
pls.default <- function(object, holdout=NULL, ...){
    if(is.null(holdout)){
        stop("We need the values from the holdout in order to proceed.",
             call.=FALSE);
    }
    h <- length(holdout);
    yForecast <- forecast(object, h=h)$mean;

    return(sum(dnorm(holdout,yForecast,sigma(object),log=TRUE)));
}

#' @rdname pls
#' @aliases pls.smooth
#' @export
pls.smooth <- function(object, holdout=NULL, ...){
    if(any(class(object)=="smoothC")){
        stop("Sorry, but PLS is not available for the combinations.",
             call.=FALSE)
    }
    # If holdout is provided, check it and use it. Otherwise try extracting from the model
    yForecast <- object$forecast;
    covarMat <- covar(object, ...);
    if(!is.null(holdout)){
        if(length(yForecast)!=length(holdout)){
            if(is.null(object$holdout)){
                stop("The forecast of the model does not correspond to the provided holdout.",
                     call.=FALSE);
            }
            else{
                holdout <- object$holdout;
            }
        }
    }
    else{
        if(all(is.na(object$holdout))){
            stop("No values for the holdout are available. Cannot proceed.",
                 call.=FALSE);
        }
        holdout <- object$holdout;
    }
    h <- length(holdout);

    Etype <- errorType(object);
    cfType <- object$cfType;
    if(any(cfType==c("MAE","MAEh","TMAE","GTMAE","MACE"))){
        cfType <- "MAE";
    }
    else if(any(cfType==c("HAM","HAMh","THAM","GTHAM","CHAM"))){
        cfType <- "HAM";
    }
    else{
        cfType <- "MSE";
    }

    densityFunction <- function(cfType, ...){
        if(cfType=="MAE"){
        # This is a simplification. The real multivariate Laplace is bizarre!
            b <- sqrt(diag(covarMat)/2);
            plsValue <- sum(dlaplace(errors, 0, b, log=TRUE));
        }
        else if(cfType=="HAM"){
        # This is a simplification. We don't have multivariate HAM yet.
            b <- (diag(covarMat)/120)^0.25;
            plsValue <- sum(ds(errors, 0, b, log=TRUE));
        }
        else{
            if(is.infinite(det(covarMat))){
                plsValue <- -as.vector((log(2*pi)+(abs(determinant(covarMat)$modulus)))/2 +
                                           (t(errors) %*% solve(covarMat) %*% errors) / 2);
            }
            else{
                # Here and later in the code the abs() is needed for weird cases of wrong covarMat
                plsValue <- -as.vector(log(2*pi*abs(det(covarMat)))/2 +
                                           (t(errors) %*% solve(covarMat) %*% errors) / 2);
            }
        }
        return(plsValue);
    }

    # Additive models
    if(Etype=="A"){
        # Non-intermittent data
        if(is.null(object$imodel)){
            errors <- holdout - yForecast;
            plsValue <- densityFunction(cfType, errors, covarMat);
        }
        # Intermittent data
        else{
            ot <- holdout!=0;
            pForecast <- object$imodel$forecast;
            errors <- holdout - yForecast / pForecast;
            if(all(ot)){
                plsValue <- densityFunction(cfType, errors, covarMat) + sum(log(pForecast));
            }
            else if(all(!ot)){
                plsValue <- sum(log(1-pForecast));
            }
            else{
                errors[!ot] <- 0;

                plsValue <- densityFunction(cfType, errors, covarMat);
                plsValue <- plsValue + sum(log(pForecast[ot])) + sum(log(1-pForecast[!ot]));
            }
        }
    }
    # Multiplicative models
    else{
        # Non-intermittent data
        if(is.null(object$imodel)){
            errors <- log(holdout) - log(yForecast);
            plsValue <- densityFunction(cfType, errors, covarMat) - sum(log(holdout));
        }
        # Intermittent data
        else{
            ot <- holdout!=0;
            pForecast <- object$imodel$forecast;
            errors <- log(holdout) - log(yForecast / pForecast);
            if(all(ot)){
                plsValue <- (densityFunction(cfType, errors, covarMat) - sum(log(holdout)) +
                             sum(log(pForecast)));
            }
            else if(all(!ot)){
                plsValue <- sum(log(1-pForecast));
            }
            else{
                errors[!ot] <- 0;

                plsValue <- densityFunction(cfType, errors, covarMat) - sum(log(holdout[ot]));
                plsValue <- plsValue + sum(log(pForecast[ot])) + sum(log(1-pForecast[!ot]));
            }
        }
    }

    return(plsValue);
}

#' @importFrom stats sigma
#' @export
sigma.smooth <- function(object, ...){
    return(sqrt(object$s2));
}

#' @export
sigma.ets <- function(object, ...){
    return(sqrt(object$sigma2));
}

#### pointLik for smooth ####
#' @importFrom greybox pointLik
#' @export
pointLik.smooth <- function(object, ...){
    obs <- nobs(object);
    errors <- residuals(object);
    likValues <- vector("numeric",obs);
    cfType <- object$cfType;

    if(errorType(object)=="M"){
        errors <- log(1+errors);
        likValues <- likValues - log(getResponse(object));
    }

    if(any(cfType==c("MAE","MAEh","TMAE","GTMAE","MACE"))){
        likValues <- likValues + dlaplace(errors, 0, mean(abs(errors)), TRUE);
    }
    else if(any(cfType==c("HAM","HAMh","THAM","GTHAM","CHAM"))){
        likValues <- likValues + ds(errors, 0, mean(sqrt(abs(errors))/2), TRUE);
    }
    else{
        likValues <- likValues + dnorm(errors, 0, sqrt(mean(abs(errors)^2)), TRUE);
    }

    likValues <- ts(as.vector(likValues), start=start(errors), frequency=frequency(errors));

    return(likValues);
}

#### Extraction of parameters of models ####
#' @export
coef.smooth <- function(object, ...)
{
    smoothType <- smoothType(object);
    if(smoothType=="CES"){
        parameters <- c(object$A,object$B);
    }
    else if(smoothType=="ETS"){
        if(any(unlist(gregexpr("C",modelType(object)))==-1)){
            # If this was normal ETS, return values
            parameters <- c(object$persistence,object$initial,object$initialSeason,object$initialX);
        }
        else{
            # If we did combinations, we cannot return anything
            message("Combination of models was done, so there are no coefficients to return");
            parameters <- NULL;
        }
    }
    else if(smoothType=="GUM"){
        parameters <- c(object$measurement,object$transition,object$persistence,object$initial);
        names(parameters) <- c(paste0("Measurement ",c(1:length(object$measurement))),
                               paste0("Transition ",c(1:length(object$transition))),
                               paste0("Persistence ",c(1:length(object$persistence))),
                               paste0("Initial ",c(1:length(object$initial))));
    }
    else if(smoothType=="ARIMA"){
        if(any(unlist(gregexpr("combine",object$model))==-1)){
            # If this was normal ARIMA, return values
            namesConstant <- NamesMA <- NamesAR <- parameters <- NULL;
            if(any(object$AR!=0)){
                parameters <- c(parameters,object$AR);
                NamesAR <- paste(rownames(object$AR),rep(colnames(object$AR),each=nrow(object$AR)),sep=", ");
            }
            if(any(object$MA!=0)){
                parameters <- c(parameters,object$MA);
                NamesMA <- paste(rownames(object$MA),rep(colnames(object$MA),each=nrow(object$MA)),sep=", ")
            }
            if(object$constant!=0){
                parameters <- c(parameters,object$constant);
                namesConstant <- "Constant";
            }
            names(parameters) <- c(NamesAR,NamesMA,namesConstant);
            parameters <- parameters[parameters!=0];
        }
        else{
            # If we did combinations, we cannot return anything
            message("Combination of models was done, so there are no coefficients to return");
            parameters <- NULL;
        }
    }
    else if(smoothType=="SMA"){
        parameters <- object$persistence;
    }

    return(parameters);
}

#' @importFrom forecast getResponse
#' @export
forecast::getResponse

#### Fitted, forecast and actual values ####
#' @export
fitted.smooth <- function(object, ...){
    return(object$fitted);
}

#' @importFrom forecast forecast
#' @export forecast
NULL

#' Forecasting time series using smooth functions
#'
#' This function is created in order for the package to be compatible with Rob
#' Hyndman's "forecast" package
#'
#' This is not a compulsory function. You can simply use \link[smooth]{es},
#' \link[smooth]{ces}, \link[smooth]{gum} or \link[smooth]{ssarima} without
#' \code{forecast.smooth}. But if you are really used to \code{forecast}
#' function, then go ahead!
#'
#' @aliases forecast forecast.smooth
#' @param object Time series model for which forecasts are required.
#' @param h Forecast horizon
#' @param intervals Type of intervals to construct. See \link[smooth]{es} for
#' details.
#' @param level Confidence level. Defines width of prediction interval.
#' @param ...  Other arguments accepted by either \link[smooth]{es},
#' \link[smooth]{ces}, \link[smooth]{gum} or \link[smooth]{ssarima}.
#' @return Returns object of class "smooth.forecast", which contains:
#'
#' \itemize{
#' \item \code{model} - the estimated model (ES / CES / GUM / SSARIMA).
#' \item \code{method} - the name of the estimated model (ES / CES / GUM / SSARIMA).
#' \item \code{fitted} - fitted values of the model.
#' \item \code{actuals} - actuals provided in the call of the model.
#' \item \code{forecast} aka \code{mean} - point forecasts of the model
#' (conditional mean).
#' \item \code{lower} - lower bound of prediction intervals.
#' \item \code{upper} - upper bound of prediction intervals.
#' \item \code{level} - confidence level.
#' \item \code{intervals} - binary variable (whether intervals were produced or not).
#' \item \code{residuals} - the residuals of the original model.
#' }
#' @template ssAuthor
#' @seealso \code{\link[forecast]{ets}, \link[forecast]{forecast}}
#' @references Hyndman, R.J., Koehler, A.B., Ord, J.K., and Snyder, R.D. (2008)
#' Forecasting with exponential smoothing: the state space approach,
#' Springer-Verlag. \url{http://www.exponentialsmoothing.net}.
#' @keywords ts univar
#' @examples
#'
#' ourModel <- ces(rnorm(100,0,1),h=10)
#'
#' forecast.smooth(ourModel,h=10)
#' forecast.smooth(ourModel,h=10,intervals=TRUE)
#' plot(forecast.smooth(ourModel,h=10,intervals=TRUE))
#'
#' @export forecast.smooth
#' @export
forecast.smooth <- function(object, h=10,
                            intervals=c("parametric","semiparametric","nonparametric","none"),
                            level=0.95, ...){
    smoothType <- smoothType(object);
    intervals <- intervals[1];
    if(smoothType=="ETS"){
        newModel <- es(object$actuals,model=object,h=h,intervals=intervals,level=level,silent="all",...);
    }
    else if(smoothType=="CES"){
        newModel <- ces(object$actuals,model=object,h=h,intervals=intervals,level=level,silent="all",...);
    }
    else if(smoothType=="GUM"){
        newModel <- gum(object$actuals,model=object,type=errorType(object),h=h,intervals=intervals,level=level,silent="all",...);
    }
    else if(smoothType=="ARIMA"){
        if(any(unlist(gregexpr("combine",object$model))==-1)){
            if(any(class(object)=="msarima")){
                newModel <- msarima(object$actuals,model=object,h=h,intervals=intervals,level=level,silent="all",...);
            }
            else{
                newModel <- ssarima(object$actuals,model=object,h=h,intervals=intervals,level=level,silent="all",...);
            }
        }
        else{
            stop(paste0("Sorry, but in order to produce forecasts for this ARIMA we need to recombine it.\n",
                 "You will have to use auto.ssarima() function instead."),call.=FALSE);
        }
    }
    else if(smoothType=="SMA"){
        newModel <- sma(object$actuals,model=object,h=h,intervals=intervals,level=level,silent="all",...);
    }
    else{
        stop("Wrong object provided. This needs to be either 'ETS', or 'CES', or 'GUM', or 'SSARIMA', or 'SMA' model.",call.=FALSE);
    }
    output <- list(model=object,method=object$model,fitted=newModel$fitted,actuals=newModel$actuals,
                   forecast=newModel$forecast,lower=newModel$lower,upper=newModel$upper,level=newModel$level,
                   intervals=intervals,mean=newModel$forecast,x=object$actuals,residuals=object$residuals);

    return(structure(output,class=c("smooth.forecast","forecast")));
}

#' @importFrom stats window
#' @export
getResponse.smooth <- function(object, ...){
    return(window(object$actuals,start(object$actuals),end(object$fitted)));
}
#' @export
getResponse.smooth.forecast <- function(object, ...){
    return(window(object$model$actuals,start(object$model$actuals),end(object$model$fitted)));
}

#### Function extracts lags of provided model ####
#' @export
lags.default <- function(object, ...){
    return(NA);
}

#' @export
lags.ets <- function(object, ...){
    modelName <- modelType(object);
    lags <- c(1);
    if(substr(modelName,nchar(modelName),nchar(modelName))!="N"){
        lags <- c(lags,frequency(getResponse(object)));
    }
    return(lags);
}

#' @export
lags.ar <- function(object, ...){
    return(1);
}

#' @export
lags.Arima <- function(object, ...){
    return(c(1,object$arma[5]));
}

#' @export
lags.smooth <- function(object, ...){
    model <- object$model;
    smoothType <- smoothType(object);
    if(!is.null(model)){
        if(smoothType=="GUM"){
            lags <- as.numeric(substring(model,unlist(gregexpr("\\[",model))+1,unlist(gregexpr("\\]",model))-1));
        }
        else if(smoothType=="ARIMA"){
            if(any(unlist(gregexpr("combine",object$model))==-1)){
                if(any(unlist(gregexpr("\\[",model))!=-1)){
                    lags <- as.numeric(substring(model,unlist(gregexpr("\\[",model))+1,unlist(gregexpr("\\]",model))-1));
                }
                else{
                    lags <- 1;
                }
            }
            else{
                warning("ARIMA was combined and we cannot extract lags anymore. Sorry!",call.=FALSE);
            }
        }
        else if(smoothType=="SMA"){
            lags <- 1;
        }
        else if(smoothType=="ETS"){
            modelName <- modelType(object);
            lags <- c(1);
            if(substr(modelName,nchar(modelName),nchar(modelName))!="N"){
                lags <- c(lags,frequency(getResponse(object)));
            }
        }
        else if(smoothType=="CES"){
            modelName <- modelType(object);
            dataFreq <- frequency(getResponse(object));
            if(modelName=="none"){
                lags <- c(1);
            }
            else if(modelName=="simple"){
                lags <- c(dataFreq);
            }
            else if(modelName=="partial"){
                lags <- c(1,dataFreq);
            }
            else if(modelName=="full"){
                lags <- c(1,dataFreq);
            }
            else{
                stop("Sorry, but we cannot identify the type of the provided model.",
                     call.=FALSE);
            }
        }
        else{
            lags <- NA;
        }
    }
    else{
        lags <- NA;
    }

    return(lags);
}

#' @export
lags.smooth.sim <- lags.smooth;

#### Function extracts type of error in the model: "A" or "M" ####
#' @importFrom greybox errorType
#' @export
errorType.smooth <- function(object, ...){
    smoothType <- smoothType(object);
    # ETS models
    if(smoothType=="ETS"){
        if(any(substr(modelType(object),1,1)==c("A","X"))){
            Etype <- "A";
        }
        else if(any(substr(modelType(object),1,1)==c("M","Y"))){
            Etype <- "M";
        }
        else{
            stop("Sorry, but we cannot define error type for this type of model",
                 call.=FALSE);
        }
    }
    # GUM models
    else if(smoothType=="GUM"){
        if(substr(modelName(object),1,1)=="M"){
            Etype <- "M";
        }
        else{
            Etype <- "A";
        }
    }
    # SSARIMA models
    else if(smoothType=="ARIMA"){
        Etype <- "A";
    }
    # CES models
    else if(smoothType=="CES"){
        Etype <- "A";
    }
    # SMA models
    else if(smoothType=="SMA"){
        Etype <- "A";
    }
    # SMA models
    else if(smoothType=="CMA"){
        Etype <- "A";
    }
    else{
        stop(paste0("Sorry but we cannot identify error type for the model '",object$model),
             call.=FALSE);
    }
    return(Etype);
}

#' @export
errorType.smooth.sim <- errorType.smooth;

#' @export
errorType.iss <- function(object, ...){
    return(substr(modelType(object),1,1));
}

##### Function returns the modellags from the model - internal function #####
modelLags.default <- function(object, ...){
    modelLags <- NA;
    if(any(class(object)=="msarima")){
        modelLags <- object$modelLags;
    }
    else{
        smoothType <- smoothType(object);
        if(smoothType=="ETS"){
            modelLags <- matrix(rep(lags(object),times=orders(object)),ncol=1);
        }
        else if(smoothType=="GUM"){
            modelLags <- matrix(rep(lags(object),times=orders(object)),ncol=1);
        }
        else if(smoothType=="ARIMA"){
            ordersARIMA <- orders(object);
            nComponents <- max(ordersARIMA$ar %*% lags(object) + ordersARIMA$i %*% lags(object),
                               ordersARIMA$ma %*% lags(object));
            modelLags <- matrix(rep(1,times=nComponents),ncol=1);
            if(is.numeric(object$constant)){
                modelLags <- rbind(modelLags,1);
            }
        }
        else if(smoothType=="CES"){
            modelLags <- matrix(rep(lags(object),times=orders(object)),ncol=1);
        }
        else if(smoothType=="SMA"){
            modelLags <- matrix(rep(1,times=orders(object)),ncol=1);
        }
    }
    return(modelLags);
}

#### Function extracts the full name of a model. For example "ETS(AAN)" ####
#' @export
modelName.default <- function(object, ...){
    return(NA);
}

#' @export
modelName.ar <- function(object, ...){
    return(paste0("AR(",object$order,")"));
}

#' @export
modelName.lm <- function(object, ...){
    return("Regression");
}

#' @export
modelName.Arima <- function(object, ...){
    ordersArma <- object$arma;
    if(all(ordersArma[c(3,4,7)]==0)){
        return(paste0("ARIMA(",paste(ordersArma[c(1,6,2)],collapse=","),")"));
    }
    else{
        return(paste0("SARIMA(",paste(ordersArma[c(1,6,2)],collapse=","),")(",
                      paste(ordersArma[c(3,7,4)],collapse=","),")[",ordersArma[5],"]"));
    }
}

#' @export
modelName.ets <- function(object, ...){
    return(object$method);
}

#' @export
modelName.forecast <- function(object, ...){
    return(object$method);
}

#' @export
modelName.smooth <- function(object, ...){
    return(object$model);
}

#### Function extracts type of model. For example "AAN" from ets ####
#' @export
modelType.default <- function(object, ...){
    modelType <- NA;
    if(is.null(object$model)){
        if(any(gregexpr("ets",object$call)!=-1)){
            model <- object$method;
            modelType <- gsub(",","",substring(model,5,nchar(model)-1));
        }
    }
    return(modelType);
}

#' @export
modelType.smooth <- function(object, ...){
    model <- object$model;
    smoothType <- smoothType(object);

    if(smoothType=="ETS"){
        modelType <- substring(model,unlist(gregexpr("\\(",model))+1,unlist(gregexpr("\\)",model))-1);
    }
    else if(smoothType=="CES"){
        modelType <- substring(model,unlist(gregexpr("\\(",model))+1,unlist(gregexpr("\\)",model))-1);
        if(modelType=="n"){
            modelType <- "none";
        }
        else if(modelType=="s"){
            modelType <- "simple";
        }
        else if(modelType=="p"){
            modelType <- "partial";
        }
        else{
            modelType <- "full";
        }
    }
    else{
        modelType <- NA;
    }

    return(modelType);
}

#' @export
modelType.smooth.sim <- modelType.smooth;

#' @export
modelType.iss <- function(object, ...){
    return(object$model);
}

#### Function extracts orders of provided model ####
#' @export
orders.default <- function(object, ...){
    return(NA);
}

#' @export
orders.smooth <- function(object, ...){
    smoothType <- smoothType(object);
    model <- object$model;
    if(!is.null(model)){
        if(smoothType=="GUM"){
            orders <- as.numeric(substring(model,unlist(gregexpr("\\[",model))-1,unlist(gregexpr("\\[",model))-1));
        }
        else if(smoothType=="ARIMA"){
            if(any(unlist(gregexpr("combine",object$model))==-1)){
                arima.orders <- paste0(c("",substring(model,unlist(gregexpr("\\(",model))+1,unlist(gregexpr("\\)",model))-1),"")
                                       ,collapse=";");
                comas <- unlist(gregexpr("\\,",arima.orders));
                semicolons <- unlist(gregexpr("\\;",arima.orders));
                ar.orders <- as.numeric(substring(arima.orders,semicolons[-length(semicolons)]+1,comas[2*(1:(length(comas)/2))-1]-1));
                i.orders <- as.numeric(substring(arima.orders,comas[2*(1:(length(comas)/2))-1]+1,comas[2*(1:(length(comas)/2))-1]+1));
                ma.orders <- as.numeric(substring(arima.orders,comas[2*(1:(length(comas)/2))]+1,semicolons[-1]-1));

                orders <- list(ar=ar.orders,i=i.orders,ma=ma.orders);
            }
            else{
                warning("ARIMA was combined and we cannot extract orders anymore. Sorry!",call.=FALSE);
            }
        }
        else if(smoothType=="SMA"){
            orders <- as.numeric(substring(model,unlist(gregexpr("\\(",model))+1,unlist(gregexpr("\\)",model))-1));
        }
        else if(smoothType=="ETS"){
            modelName <- modelType(object);
            orders <- 1;
            if(substr(modelName,2,2)!="N"){
                orders <- 2;
            }
            if(substr(modelName,nchar(modelName),nchar(modelName))!="N"){
                orders <- c(orders, 1);
            }
        }
        else if(smoothType=="CES"){
            modelName <- modelType(object);
            if(modelName=="none"){
                orders <- 2;
            }
            else if(modelName=="simple"){
                orders <- 2;
            }
            else if(modelName=="partial"){
                orders <- c(2,1);
            }
            else if(modelName=="full"){
                orders <- c(2,2);
            }
            else{
                stop("Sorry, but we cannot identify the type of the provided model.",
                     call.=FALSE);
            }
        }
        else{
            orders <- NA;
        }
    }
    else{
        orders <- NA;
    }

    return(orders);
}

#' @export
orders.smooth.sim <- orders.smooth;

#' @export
orders.ar <- function(object, ...){
    return(list(ar=object$order));
}

#' @export
orders.Arima <- function(object, ...){
    model <- object$arma;

    ar.orders <- c(model[1],model[3]);
    i.orders <-  c(model[6],model[7]);
    ma.orders <- c(model[2],model[4]);

    orders <- list(ar=ar.orders,i=i.orders,ma=ma.orders);

    return(orders);
}

#### Plots of smooth objects ####
#' @importFrom graphics plot
#' @export
plot.smooth <- function(x, ...){
    ellipsis <- list(...);
    parDefault <- par(no.readonly = TRUE);
    smoothType <- smoothType(x);
    if(smoothType=="ETS"){
        if(any(unlist(gregexpr("C",x$model))==-1)){
            if(ncol(x$states)>10){
                message("Too many states. Plotting them one by one on several graphs.");
                if(is.null(ellipsis$main)){
                    ellipsisMain <- NULL;
                }
                else{
                    ellipsisMain <- ellipsis$main;
                }
                nPlots <- ceiling(ncol(x$states)/10);
                for(i in 1:nPlots){
                    if(is.null(ellipsisMain)){
                        ellipsis$main <- paste0("States of ",x$model,", part ",i);
                    }
                    ellipsis$x <- x$states[,(1+(i-1)*10):min(i*10,ncol(x$states))];
                    do.call(plot, ellipsis);
                }
            }
            else{
                if(is.null(ellipsis$main)){
                    ellipsis$main <- paste0("States of ",x$model);
                }
                ellipsis$x <- x$states;
                do.call(plot, ellipsis);
            }
        }
        else{
            # If we did combinations, we cannot return anything
            message("Combination of models was done. Sorry, but there is nothing to plot.");
        }
    }
    else if(smoothType=="CMA"){
        ellipsis$actuals <- x$actuals;
        ellipsis$forecast <- x$forecast;
        ellipsis$fitted <- x$fitted;
        ellipsis$legend <- FALSE;
        ellipsis$vline <- FALSE;
        if(is.null(ellipsis$main)){
            ellipsis$main <- x$model;
        }
        do.call(graphmaker, ellipsis);
    }
    else{
        if(any(unlist(gregexpr("combine",x$model))!=-1)){
            # If we did combinations, we cannot do anything
            message("Combination of models was done. Sorry, but there is nothing to plot.");
        }
        else{
            if(ncol(x$states)>10){
                message("Too many states. Plotting them one by one on several graphs.");
                if(is.null(ellipsis$main)){
                    ellipsisMain <- NULL;
                }
                else{
                    ellipsisMain <- ellipsis$main;
                }
                nPlots <- ceiling(ncol(x$states)/10);
                for(i in 1:nPlots){
                    if(is.null(ellipsisMain)){
                        ellipsis$main <- paste0("States of ",x$model,", part ",i);
                    }
                    ellipsis$x <- x$states[,(1+(i-1)*10):min(i*10,ncol(x$states))];
                    do.call(plot, ellipsis);
                }
            }
            else{
                if(is.null(ellipsis$main)){
                    ellipsis$main <- paste0("States of ",x$model);
                }
                ellipsis$x <- x$states;
                do.call(plot, ellipsis);
            }
        }
    }
    par(parDefault);
}

#' @export
plot.smoothC <- function(x, ...){
    graphmaker(x$actuals, x$forecast, x$fitted, x$lower, x$upper, x$level,
               main="Combined smooth forecasts");
}

#' @export
plot.smooth.sim <- function(x, ...){
    ellipsis <- list(...);
    if(is.null(ellipsis$main)){
        ellipsis$main <- x$model;
    }

    if(is.null(dim(x$data))){
        nsim <- 1;
    }
    else{
        nsim <- dim(x$data)[2];
    }

    if(nsim==1){
        if(is.null(ellipsis$ylab)){
            ellipsis$ylab <- "Data";
        }
        ellipsis$x <- x$data;
        do.call(plot, ellipsis);
    }
    else{
        randomNumber <- ceiling(runif(1,1,nsim));
        message(paste0("You have generated ",nsim," time series. Not sure which of them to plot.\n",
                       "Please use plot(ourSimulation$data[,k]) instead. Plotting randomly selected series N",randomNumber,"."));
        if(is.null(ellipsis$ylab)){
            ellipsis$ylab <- paste0("Series N",randomNumber);
        }
        ellipsis$x <- x$data[,randomNumber];
        do.call(plot, ellipsis);
    }
}

#' @method plot smooth.forecast
#' @export
plot.smooth.forecast <- function(x, ...){
    if(any(x$intervals!=c("none","n"))){
        graphmaker(x$actuals,x$forecast,x$fitted,x$lower,x$upper,x$level,main=x$method);
    }
    else{
        graphmaker(x$actuals,x$forecast,x$fitted,main=x$method);
    }
}

#' @export
plot.iss <- function(x, ...){
    ellipsis <- list(...);
    intermittent <- x$intermittent
    if(intermittent=="i"){
        intermittent <- "Interval-based";
    }
    else if(intermittent=="p"){
        intermittent <- "Probability-based";
    }
    else if(intermittent=="f"){
        intermittent <- "Fixed probability";
    }
    else if(intermittent=="l"){
        intermittent <- "Logistic probability";
    }
    else{
        intermittent <- "None";
    }
    if(is.null(ellipsis$main)){
        graphmaker(x$actuals,x$forecast,x$fitted,main=paste0("iSS, ",intermittent), ...);
    }
    else{
        graphmaker(x$actuals,x$forecast,x$fitted, ...);
    }
}

#### Prints of smooth ####
#' @export
print.smooth <- function(x, ...){
    smoothType <- smoothType(x);

    if(!is.list(x$model)){
        if(smoothType=="CMA"){
            holdout <- FALSE;
            intervals <- FALSE;
            cumulative <- FALSE;
        }
        else{
            holdout <- any(!is.na(x$holdout));
            intervals <- any(!is.na(x$lower));
            cumulative <- x$cumulative;
        }
    }
    else{
        holdout <- any(!is.na(x$holdout));
        intervals <- any(!is.na(x$lower));
        cumulative <- x$cumulative;
    }

    if(all(holdout,intervals)){
        if(!cumulative){
            insideintervals <- sum((x$holdout <= x$upper) & (x$holdout >= x$lower)) / length(x$forecast) * 100;
        }
        else{
            insideintervals <- NULL;
        }
    }
    else{
        insideintervals <- NULL;
    }

    intervalsType <- x$intervals;

    if(!is.null(x$model)){
        if(!is.list(x$model)){
            if(any(smoothType==c("SMA","CMA"))){
                x$iprob <- 1;
                x$initialType <- "b";
                intermittent <- "n";
            }
            else if(smoothType=="ETS"){
                # If cumulative forecast and Etype=="M", report that this was "parameteric" interval
                if(cumulative & substr(modelType(x),1,1)=="M"){
                    intervalsType <- "p";
                }
            }
        }
    }
    if(class(x$imodel)!="iss"){
        intermittent <- "n";
    }
    else{
        intermittent <- x$imodel$intermittent;
    }

    ssOutput(x$timeElapsed, x$model, persistence=x$persistence, transition=x$transition, measurement=x$measurement,
             phi=x$phi, ARterms=x$AR, MAterms=x$MA, constant=x$constant, A=x$A, B=x$B,initialType=x$initialType,
             nParam=x$nParam, s2=x$s2, hadxreg=!is.null(x$xreg), wentwild=x$updateX,
             cfType=x$cfType, cfObjective=x$cf, intervals=intervals, cumulative=cumulative,
             intervalsType=intervalsType, level=x$level, ICs=x$ICs,
             holdout=holdout, insideintervals=insideintervals, errormeasures=x$accuracy,
             intermittent=intermittent);
}

#' @export
print.smooth.sim <- function(x, ...){
    smoothType <- smoothType(x);
    if(is.null(dim(x$data))){
        nsim <- 1
    }
    else{
        nsim <- dim(x$data)[2]
    }

    cat(paste0("Data generated from: ",x$model,"\n"));
    cat(paste0("Number of generated series: ",nsim,"\n"));

    if(nsim==1){
        if(smoothType=="ETS"){
            cat(paste0("Persistence vector: \n"));
            xPersistence <- as.vector(x$persistence);
            names(xPersistence) <- rownames(x$persistence);
            print(round(xPersistence,3));
            if(x$phi!=1){
                cat(paste0("Phi: ",x$phi,"\n"));
            }
            if(x$intermittent!="n"){
                cat(paste0("Intermittence type: ",x$intermittent,"\n"));
            }
            cat(paste0("True likelihood: ",round(x$logLik,3),"\n"));
        }
        else if(smoothType=="ARIMA"){
            ar.orders <- orders(x)$ar;
            i.orders <- orders(x)$i;
            ma.orders <- orders(x)$ma;
            lags <- lags(x);
            # AR terms
            if(any(ar.orders!=0)){
                ARterms <- matrix(0,max(ar.orders),sum(ar.orders!=0),
                                  dimnames=list(paste0("AR(",c(1:max(ar.orders)),")"),
                                                paste0("Lag ",lags[ar.orders!=0])));
            }
            else{
                ARterms <- matrix(0,1,1);
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
                MAterms <- matrix(0,1,1);
            }

            n.coef <- ar.coef <- ma.coef <- 0;
            ar.i <- ma.i <- 1;
            for(i in 1:length(ar.orders)){
                if(ar.orders[i]!=0){
                    ARterms[1:ar.orders[i],ar.i] <- x$AR[ar.coef+(1:ar.orders[i])];
                    ar.coef <- ar.coef + ar.orders[i];
                    ar.i <- ar.i + 1;
                }
                if(ma.orders[i]!=0){
                    MAterms[1:ma.orders[i],ma.i] <- x$MA[ma.coef+(1:ma.orders[i])];
                    ma.coef <- ma.coef + ma.orders[i];
                    ma.i <- ma.i + 1;
                }
            }

            if(!is.null(x$AR)){
                cat(paste0("AR parameters: \n"));
                print(round(ARterms,3));
            }
            if(!is.null(x$MA)){
                cat(paste0("MA parameters: \n"));
                print(round(MAterms,3));
            }
            if(!is.na(x$constant)){
                cat(paste0("Constant value: ",round(x$constant,3),"\n"));
            }
            cat(paste0("True likelihood: ",round(x$logLik,3),"\n"));
        }
        else if(smoothType=="CES"){
            cat(paste0("Smoothing parameter A: ",round(x$A,3),"\n"));
            if(!is.null(x$B)){
                if(is.complex(x$B)){
                    cat(paste0("Smoothing parameter B: ",round(x$B,3),"\n"));
                }
                else{
                    cat(paste0("Smoothing parameter b: ",round(x$B,3),"\n"));
                }
            }
            cat(paste0("True likelihood: ",round(x$logLik,3),"\n"));
        }
        else if(smoothType=="SMA"){
            cat(paste0("True likelihood: ",round(x$logLik,3),"\n"));
        }
    }
}

#' @export
print.smooth.forecast <- function(x, ...){
    if(any(x$intervals!=c("none","n"))){
        level <- x$level;
        if(level>1){
            level <- level/100;
        }
        output <- cbind(x$forecast,x$lower,x$upper);
        colnames(output) <- c("Point forecast",paste0("Lower bound (",(1-level)/2*100,"%)"),paste0("Upper bound (",(1+level)/2*100,"%)"));
    }
    else{
        output <- x$forecast;
    }
    print(output);
}

#' @export
print.iss <- function(x, ...){
    intermittent <- x$intermittent
    if(intermittent=="i"){
        intermittent <- "Interval-based";
    }
    else if(intermittent=="p"){
        intermittent <- "Probability-based";
    }
    else if(intermittent=="f"){
        intermittent <- "Fixed probability";
    }
    else if(intermittent=="l"){
        intermittent <- "Logistic probability";
    }
    else if(intermittent=="s"){
        intermittent <- "Interval-based with SBA correction";
    }
    else{
        intermittent <- "None";
    }
    ICs <- round(c(AIC(x),AICc(x),BIC(x),BICc(x)),4);
    names(ICs) <- c("AIC","AICc","BIC","BICc");
    cat(paste0("Intermittent state space model estimated: ",intermittent,"\n"));
    if(!is.null(x$model)){
        cat(paste0("Underlying ETS model: ",x$model,"\n"));
    }
    if(!is.null(x$persistence)){
        cat("Smoothing parameters:\n");
        print(round(x$persistence,3));
    }
    if(!is.null(x$initial)){
        cat("Vector of initials:\n");
        print(round(x$initial,3));
    }
    # cat(paste0("Probability forecast: ",round(x$forecast[1],3),"\n"));
    cat("Information criteria: \n");
    print(ICs);
}

#### Simulate data using provided object ####
#' @export
simulate.smooth <- function(object, nsim=1, seed=NULL, obs=NULL, ...){
    ellipsis <- list(...);
    smoothType <- smoothType(object);
    if(is.null(obs)){
        obs <- nobs(object);
    }
    if(!is.null(seed)){
        set.seed(seed);
    }

    # Start a list of arguments
    args <- vector("list",0);

    cfType <- object$cfType;
    if(any(cfType==c("MAE","MAEh","TMAE","GTMAE","MACE"))){
        randomizer <- "rlaplace";
        if(!is.null(ellipsis$mu)){
            args$mu <- ellipsis$mu;
        }
        else{
            args$mu <- 0;
        }

        if(!is.null(ellipsis$b)){
            args$b <- ellipsis$b;
        }
        else{
            args$b <- mean(abs(residuals(object)));
        }
    }
    else if(any(cfType==c("HAM","HAMh","THAM","GTHAM","CHAM"))){
        randomizer <- "rs";
        if(!is.null(ellipsis$mu)){
            args$mu <- ellipsis$mu;
        }
        else{
            args$mu <- 0;
        }

        if(!is.null(ellipsis$b)){
            args$b <- ellipsis$b;
        }
        else{
            args$b <- mean(sqrt(abs(residuals(object))));
        }
    }
    else{
        if(errorType(object)=="A"){
            randomizer <- "rnorm";
        }
        else{
            randomizer <- "rlnorm";
        }
        if(!is.null(ellipsis$mean)){
            args$mean <- ellipsis$mean;
        }
        else{
            args$mean <- 0;
        }

        if(!is.null(ellipsis$sd)){
            args$sd <- ellipsis$sd;
        }
        else{
            args$sd <- sigma(object);
        }
    }
    args$randomizer <- randomizer;
    args$frequency <- frequency(object$actuals);
    args$obs <- obs;
    args$nsim <- nsim;
    args$initial <- object$initial;
    args$iprob <- object$iprob[length(object$iprob)];

    if(smoothType=="ETS"){
        model <- modelType(object);
        if(any(unlist(gregexpr("C",model))==-1)){
            args <- c(args,list(model=model, phi=object$phi, persistence=object$persistence,
                                initialSeason=object$initialSeason));

            simulatedData <- do.call("sim.es",args);
        }
        else{
            message("Sorry, but we cannot simulate data from combined model.");
            simulatedData <- NA;
        }
    }
    else if(smoothType=="ARIMA"){
        if(any(unlist(gregexpr("combine",object$model))==-1)){
            args <- c(args,list(orders=orders(object), lags=lags(object),
                                AR=object$AR, MA=object$MA, constant=object$constant));

            simulatedData <- do.call("sim.ssarima",args);
        }
        else{
            message("Sorry, but we cannot simulate data from combined model.");
            simulatedData <- NA;
        }
    }
    else if(smoothType=="CES"){
        args <- c(args,list(seasonality=modelType(object), A=object$A, B=object$B));

        simulatedData <- do.call("sim.ces",args);
    }
    else if(smoothType=="GUM"){
        args <- c(args,list(orders=orders(object), lags=lags(object),
                            measurement=object$measurement, transition=object$transition,
                            persistence=object$persistence));

        simulatedData <- do.call("sim.gum",args);
    }
    else if(smoothType=="SMA"){
        args <- c(args,list(order=orders(object)));

        simulatedData <- do.call("sim.sma",args);
    }
    else{
        model <- substring(object$model,1,unlist(gregexpr("\\(",object$model))[1]-1);
        message(paste0("Sorry, but simulate is not yet available for the model ",model,"."));
        simulatedData <- NA;
    }
    return(simulatedData);
}

#### Type of smooth model. Internal function ####
smoothType.default <- function(object, ...){
    return(NA);
}

smoothType.smooth <- function(object, ...){
    if(!is.list(object$model)){
        if(gregexpr("ETS",object$model)!=-1){
            smoothType <- "ETS";
        }
        else if(gregexpr("CES",object$model)!=-1){
            smoothType <- "CES";
        }
        else if(gregexpr("ARIMA",object$model)!=-1){
            smoothType <- "ARIMA";
        }
        else if(gregexpr("GUM",object$model)!=-1){
            smoothType <- "GUM";
        }
        else if(gregexpr("SMA",object$model)!=-1){
            smoothType <- "SMA";
        }
        else if(gregexpr("CMA",object$model)!=-1){
            smoothType <- "CMA";
        }
        else if(gregexpr("VES",object$model)!=-1){
            smoothType <- "VES";
        }
        else{
            smoothType <- NA;
        }
    }
    else{
        smoothType <- "smoothCombine";
    }

    return(smoothType);
}

smoothType.smooth.sim <- smoothType.smooth;

#### Summary of objects ####
#' @export
summary.smooth <- function(object, ...){
    print(object);
}

#' @export
summary.smooth.forecast <- function(object, ...){
    print(object);
}

#' @export
summary.iss <- function(object, ...){
    print(object);
}
