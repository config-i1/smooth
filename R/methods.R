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
    nParamAll <- nparam(object);
    llikelihood <- llikelihood[1:length(llikelihood)];

    if(!is.null(object$occurrence)){
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
    nParamAll <- nparam(object);
    llikelihood <- llikelihood[1:length(llikelihood)];

    if(!is.null(object$occurrence)){
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

    if(is.smoothC(object)){
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

    if(!is.null(object$occurrence) & type=="e"){
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
        if(!is.null(object$occurrence)){
            obs <- t((errors!=0)*1) %*% (errors!=0)*1;
            obs[obs==0] <- 1;
            df <- obs - nparam(object);
            df[df<=0] <- obs[df<=0];
        }
        else{
            obs <- matrix(nobs(object),ncol(errors),ncol(errors));
            df <- obs - nparam(object);
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
            obs <- length(actuals(object));
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
            if(!is.null(object$occurrence)){
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
        if(!is.null(object$occurrence)){
            ot <- (residuals(object)!=0)*1;
        }
        else{
            ot <- rep(1,nobs(object));
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
        return(structure(object$logLik,nobs=nobs(object),df=nparam(object),class="logLik"));
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
        return(structure(object$logLik,nobs=obs,df=nparam(object),class="logLik"));
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

#' @importFrom greybox nparam

#' @export
nparam.smooth <- function(object, ...){
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
nparam.iss <- function(object, ...){
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
#' ourModel <- es(x, h=10, holdout=TRUE, interval=TRUE)
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
    if(is.smoothC(object)){
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
    loss <- object$loss;
    if(any(loss==c("MAE","MAEh","TMAE","GTMAE","MACE"))){
        loss <- "MAE";
    }
    else if(any(loss==c("HAM","HAMh","THAM","GTHAM","CHAM"))){
        loss <- "HAM";
    }
    else{
        loss <- "MSE";
    }

    densityFunction <- function(loss, ...){
        if(loss=="MAE"){
        # This is a simplification. The real multivariate Laplace is bizarre!
            scale <- sqrt(diag(covarMat)/2);
            plsValue <- sum(dlaplace(errors, 0, scale, log=TRUE));
        }
        else if(loss=="HAM"){
        # This is a simplification. We don't have multivariate HAM yet.
            scale <- (diag(covarMat)/120)^0.25;
            plsValue <- sum(ds(errors, 0, scale, log=TRUE));
        }
        else{
            if(is.infinite(det(covarMat))){
                plsValue <- -as.vector((log(2*pi)+(abs(determinant(covarMat)$modulus)))/2 +
                                           (t(errors) %*% solve(covarMat) %*% errors) / 2);
            }
            # This is the case with overfitting the data
            else if(det(covarMat)==0){
                # If there is any non-zero error, then it means that the model is completely wrong (because it predicts that sigma=0)
                if(any(errors!=0)){
                    plsValue <- -Inf;
                }
                else{
                    plsValue <- 0;
                }
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
        if(is.null(object$occurrence)){
            errors <- holdout - yForecast;
            plsValue <- densityFunction(loss, errors, covarMat);
        }
        # Intermittent data
        else{
            ot <- holdout!=0;
            pForecast <- object$occurrence$forecast;
            errors <- holdout - yForecast / pForecast;
            if(all(ot)){
                plsValue <- densityFunction(loss, errors, covarMat) + sum(log(pForecast));
            }
            else if(all(!ot)){
                plsValue <- sum(log(1-pForecast));
            }
            else{
                errors[!ot] <- 0;

                plsValue <- densityFunction(loss, errors, covarMat);
                plsValue <- plsValue + sum(log(pForecast[ot])) + sum(log(1-pForecast[!ot]));
            }
        }
    }
    # Multiplicative models
    else{
        # Non-intermittent data
        if(is.null(object$occurrence)){
            errors <- log(holdout) - log(yForecast);
            plsValue <- densityFunction(loss, errors, covarMat) - sum(log(holdout));
        }
        # Intermittent data
        else{
            ot <- holdout!=0;
            pForecast <- object$occurrence$forecast;
            errors <- log(holdout) - log(yForecast / pForecast);
            if(all(ot)){
                plsValue <- (densityFunction(loss, errors, covarMat) - sum(log(holdout)) +
                             sum(log(pForecast)));
            }
            else if(all(!ot)){
                plsValue <- sum(log(1-pForecast));
            }
            else{
                errors[!ot] <- 0;

                plsValue <- densityFunction(loss, errors, covarMat) - sum(log(holdout[ot]));
                plsValue <- plsValue + sum(log(pForecast[ot])) + sum(log(1-pForecast[!ot]));
            }
        }
    }

    return(plsValue);
}

#' @importFrom stats sigma
#' @export
sigma.smooth <- function(object, ...){
    if(!is.null(object$s2)){
        return(sqrt(object$s2));
    }
    else{
        return(NULL);
    }
}

#' @export
sigma.smooth.sim <- function(object, ...){
    return(sqrt(mean(residuals(object)^2)));
}

#### pointLik for smooth ####
#' @importFrom greybox pointLik
#' @export
pointLik.smooth <- function(object, ...){
    obs <- nobs(object);
    errors <- residuals(object);
    likValues <- vector("numeric",obs);
    loss <- object$loss;

    if(errorType(object)=="M"){
        likValues <- likValues - log(actuals(object));
    }

    if(any(loss==c("MAE","MAEh","TMAE","GTMAE","MACE"))){
        likValues <- likValues + dlaplace(errors, 0, mean(abs(errors)), TRUE);
    }
    else if(any(loss==c("HAM","HAMh","THAM","GTHAM","CHAM"))){
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


#### Fitted, forecast and actual values ####
#' @export
fitted.smooth <- function(object, ...){
    return(object$fitted);
}
#' @export
fitted.smooth.forecast <- function(object, ...){
    return(fitted(object$model));
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
#' @param interval Type of interval to construct. See \link[smooth]{es} for
#' details.
#' @param level Confidence level. Defines width of prediction interval.
#' @param ...  Other arguments accepted by either \link[smooth]{es},
#' \link[smooth]{ces}, \link[smooth]{gum} or \link[smooth]{ssarima}.
#' @return Returns object of class "smooth.forecast", which contains:
#'
#' \itemize{
#' \item \code{model} - the estimated model (ES / CES / GUM / SSARIMA).
#' \item \code{method} - the name of the estimated model (ES / CES / GUM / SSARIMA).
#' \item \code{forecast} aka \code{mean} - point forecasts of the model
#' (conditional mean).
#' \item \code{lower} - lower bound of prediction interval.
#' \item \code{upper} - upper bound of prediction interval.
#' \item \code{level} - confidence level.
#' \item \code{interval} - binary variable (whether interval were produced or not).
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
#' forecast.smooth(ourModel,h=10,interval=TRUE)
#' plot(forecast.smooth(ourModel,h=10,interval=TRUE))
#'
#' @rdname forecast.smooth
#' @export forecast.smooth
#' @export
forecast.smooth <- function(object, h=10,
                            interval=c("parametric","semiparametric","nonparametric","none"),
                            level=0.95, ...){
    smoothType <- smoothType(object);
    interval <- interval[1];
    if(smoothType=="ETS"){
        newModel <- es(actuals(object),model=object,h=h,interval=interval,level=level,silent="all",...);
    }
    else if(smoothType=="CES"){
        newModel <- ces(actuals(object),model=object,h=h,interval=interval,level=level,silent="all",...);
    }
    else if(smoothType=="GUM"){
        newModel <- gum(actuals(object),model=object,type=errorType(object),h=h,interval=interval,level=level,silent="all",...);
    }
    else if(smoothType=="ARIMA"){
        if(any(unlist(gregexpr("combine",object$model))==-1)){
            if(is.msarima(object)){
                newModel <- msarima(actuals(object),model=object,h=h,interval=interval,level=level,silent="all",...);
            }
            else{
                newModel <- ssarima(actuals(object),model=object,h=h,interval=interval,level=level,silent="all",...);
            }
        }
        else{
            stop(paste0("Sorry, but in order to produce forecasts for this ARIMA we need to recombine it.\n",
                 "You will have to use auto.ssarima() function instead."),call.=FALSE);
        }
    }
    else if(smoothType=="SMA"){
        newModel <- sma(actuals(object),model=object,h=h,interval=interval,level=level,silent="all",...);
    }
    else{
        stop("Wrong object provided. This needs to be either 'ETS', or 'CES', or 'GUM', or 'SSARIMA', or 'SMA' model.",call.=FALSE);
    }
    output <- list(model=object,method=object$model,
                   forecast=newModel$forecast,lower=newModel$lower,upper=newModel$upper,level=newModel$level,
                   interval=interval,mean=newModel$forecast);

    return(structure(output,class=c("smooth.forecast","forecast")));
}

#' @rdname forecast.smooth
#' @export
forecast.oes <- function(object, h=10,
                         interval=c("parametric","semiparametric","nonparametric","none"),
                         level=0.95, ...){
    if(is.oesg(object)){
        newModel <- oesg(actuals(object),modelA=object$modelA,modelB=object$modelB,
                         h=h,interval=interval,level=level,silent="all",...);
    }
    else{
        newModel <- oes(actuals(object),model=object,
                        h=h,interval=interval,level=level,silent="all",...);
    }

    output <- list(model=object,method=object$model,
                   forecast=newModel$forecast,lower=newModel$lower,upper=newModel$upper,level=level,
                   interval=interval,mean=newModel$forecast);

    return(structure(output,class=c("smooth.forecast","forecast")));
}

#' @importFrom stats window
#' @importFrom greybox actuals
#' @export
actuals.smooth <- function(object, ...){
    return(window(object$y,start(object$y),end(object$fitted)));
}
#' @export
actuals.smooth.forecast <- function(object, ...){
    return(window(actuals(object$model),start(actuals(object$model)),end(fitted(object$model))));
}
#' @export
actuals.iss <- function(object, ...){
    return(window(actuals(object),start(actuals(object)),end(fitted(object))));
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
        lags <- c(lags,frequency(actuals(object)));
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
                lags <- c(lags,frequency(actuals(object)));
            }
        }
        else if(smoothType=="CES"){
            modelName <- modelType(object);
            dataFreq <- frequency(actuals(object));
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

##### Function returns the modelLags from the model - internal function #####
modelLags.default <- function(object, ...){
    modelLags <- NA;
    if(is.msarima(object)){
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
    return(NA);
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

#' @export
modelType.oesg <- function(object, ...){
    return(modelType(object$modelA));
}

#' @export
modelType.ets <- function(object, ...){
    return(gsub(",","",substring(object$method,5,nchar(object$method)-1)));
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
#' Plots for the fit and states
#'
#' The function produces plot actuals, fitted values and forecasts and states of the model
#'
#' The list of produced plots includes:
#' \enumerate{
#' \item Fitted over time. Plots actuals (black line), fitted values (purple line), point forecast
#' (blue line) and prediction interval (grey lines). Can be used in order to make sure that the model
#' did not miss any important events over time;
#' \item Standardised residuals vs Fitted. Plots the points and the confidence bounds
#' (red lines) for the specified confidence \code{level}. Useful for the analysis of outliers;
#' \item Studentised residuals vs Fitted. This is similar to the previous plot, but with the
#' residuals divided by the scales with the leave-one-out approach. Should be more sensitive
#' to outliers;
#' \item Absolute residuals vs Fitted. Useful for the analysis of heteroscedasticity;
#' \item Squared residuals vs Fitted - similar to (3), but with squared values;
#' \item Q-Q plot with the specified distribution. Can be used in order to see if the
#' residuals follow the assumed distribution. The type of distribution depends on the one used
#' in the estimation (see \code{distribution} parameter in \link[greybox]{alm});
#' \item ACF of the residuals. Are the residuals autocorrelated? See \link[stats]{acf} for
#' details;
#' \item PACF of the residuals. No, really, are they autocorrelated? See \link[stats]{pacf}
#' for details;
#' \item Plot of the states of the model. It is not recommended to produce this plot together with
#' the others, because there might be several states, which would cause the creation of a different
#' canvas. In case of "msdecompose", this will produce the decomposition of the series into states
#' on a different canvas.
#' }
#' Which of the plots to produce, is specified via the \code{which} parameter.
#'
#' @param x Time series model for which forecasts are required.
#' @param which Which of the plots to produce. The possible options (see details for explanations):
#' \enumerate{
#' \item Fitted over time;
#' \item Standardised residuals vs Fitted;
#' \item Studentised residuals vs Fitted;
#' \item Absolute residuals vs Fitted;
#' \item Squared residuals vs Fitted;
#' \item Q-Q plot with the specified distribution;
#' \item ACF of the residuals;
#' \item PACF of the residuals;
#' \item Plot of states of the model.
#' }
#' @param level Confidence level. Defines width of confidence interval. Used in plots (2), (6) and
#' (7).
#' @param legend If \code{TRUE}, then the legend is produced on plots (1), (2) and (3).
#' @param ask Logical; if \code{TRUE}, the user is asked to press Enter before each plot.
#' @param lowess Logical; if \code{TRUE}, LOWESS lines are drawn on scatterplots, see \link[stats]{lowess}.
#' @param ... The parameters passed to the plot functions. Recommended to use with separate plots.
#' @return The function produces the number of plots, specified in the parameter \code{which}.
#'
#' @template ssAuthor
#' @seealso \link[greybox]{plot.greybox}
#' @keywords ts univar
#' @examples
#'
#' ourModel <- es(c(rnorm(50,100,10),rnorm(50,120,10)), "ANN", h=10)
#' par(mfcol=c(3,3))
#' plot(ourModel, c(1:9))
#'
#' @importFrom stats ppoints qqnorm qqplot qqline acf pacf lowess sd na.pass
#' @importFrom grDevices dev.interactive devAskNewPage
#' @importFrom graphics plot text
#' @rdname plot.smooth
#' @export
plot.smooth <- function(x, which=c(1,2,4,6), level=0.95, legend=FALSE,
                        ask=prod(par("mfcol")) < length(which) && dev.interactive(),
                        lowess=TRUE, ...){
    ellipsis <- list(...);

    # Define, whether to wait for the hit of "Enter"
    if(ask){
        oask <- devAskNewPage(TRUE);
        on.exit(devAskNewPage(oask));
    }

    # 1. Basic plot over time
    plot1 <- function(x, ...){
        if(any(x$interval==c("none","n"))){
            graphmaker(actuals(x), x$forecast, fitted(x), main=x$model, legend=legend, parReset=FALSE, ...);
        }
        else{
            graphmaker(actuals(x), x$forecast, fitted(x), x$lower, x$upper, x$level, main=x$model, legend=legend, parReset=FALSE, ...);
        }
    }

    # 2 and 3: Standardised  / studentised residuals vs Fitted
    plot2 <- function(x, type="rstandard", ...){
        ellipsis <- list(...);

        ellipsis$x <- as.vector(fitted(x));
        if(type=="rstandard"){
            ellipsis$y <- as.vector(rstandard(x));
            yName <- "Standardised";
        }
        else{
            ellipsis$y <- as.vector(rstudent(x));
            yName <- "Studentised";
        }

        if(is.oes(x$occurrence)){
            ellipsis$x <- ellipsis$x[actuals(x$occurrence)!=0];
            ellipsis$y <- ellipsis$y[actuals(x$occurrence)!=0];
        }

        # Remove NAs
        if(any(is.na(ellipsis$x))){
            ellipsis$x <- ellipsis$x[!is.na(ellipsis$x)];
            ellipsis$y <- ellipsis$y[!is.na(ellipsis$y)];
        }

        if(!any(names(ellipsis)=="main")){
            ellipsis$main <- paste0(yName," Residuals vs Fitted");
        }

        if(!any(names(ellipsis)=="xlab")){
            ellipsis$xlab <- "Fitted";
        }
        if(!any(names(ellipsis)=="ylab")){
            ellipsis$ylab <- paste0(yName," Residuals");
        }

        if(legend){
            if(ellipsis$x[length(ellipsis$x)]>mean(ellipsis$x)){
                legendPosition <- "bottomright";
            }
            else{
                legendPosition <- "topright";
            }
        }

        zValues <- switch(x$loss,
                          "MAE"=qlaplace(c((1-level)/2, (1+level)/2), 0, 1),
                          "HAM"=qs(c((1-level)/2, (1+level)/2), 0, 1),
                          qnorm(c((1-level)/2, (1+level)/2), 0, 1));
        outliers <- which(ellipsis$y >zValues[2] | ellipsis$y <zValues[1]);
        # cat(paste0(round(length(outliers)/length(ellipsis$y),3)*100,"% of values are outside the bounds\n"));

        if(!any(names(ellipsis)=="ylim")){
            ellipsis$ylim <- range(c(ellipsis$y,zValues), na.rm=TRUE);
            if(legend){
                if(legendPosition=="bottomright"){
                    ellipsis$ylim[1] <- ellipsis$ylim[1] - 0.2*diff(ellipsis$ylim);
                }
                else{
                    ellipsis$ylim[2] <- ellipsis$ylim[2] + 0.2*diff(ellipsis$ylim);
                }
            }
        }

        xRange <- range(ellipsis$x, na.rm=TRUE);
        xRange[1] <- xRange[1] - sd(ellipsis$x, na.rm=TRUE);
        xRange[2] <- xRange[2] + sd(ellipsis$x, na.rm=TRUE);

        do.call(plot,ellipsis);
        abline(h=0, col="grey", lty=2);
        polygon(c(xRange,rev(xRange)),c(zValues[1],zValues[1],zValues[2],zValues[2]),
                col="lightgrey", border=NA, density=10);
        abline(h=zValues, col="red", lty=2);
        if(length(outliers)>0){
            points(ellipsis$x[outliers], ellipsis$y[outliers], pch=16);
            text(ellipsis$x[outliers], ellipsis$y[outliers], labels=outliers, pos=4);
        }
        if(lowess){
            lines(lowess(ellipsis$x, ellipsis$y), col="red");
        }

        if(legend){
            if(lowess){
                legend(legendPosition,
                       legend=c(paste0(round(level,3)*100,"% bounds"),"outside the bounds","LOWESS line"),
                       col=c("red", "black","red"), lwd=c(1,NA,1), lty=c(2,1,1), pch=c(NA,16,NA));
            }
            else{
                legend(legendPosition,
                       legend=c(paste0(round(level,3)*100,"% bounds"),"outside the bounds"),
                       col=c("red", "black"), lwd=c(1,NA), lty=c(2,1), pch=c(NA,16));
            }
        }
    }

    # 4 and 5. Fitted vs |Residuals| or Fitted vs Residuals^2
    plot3 <- function(x, type="abs", ...){
        ellipsis <- list(...);

        ellipsis$x <- as.vector(fitted(x));
        if(type=="abs"){
            ellipsis$y <- abs(as.vector(residuals(x)));
        }
        else{
            ellipsis$y <- as.vector(residuals(x))^2;
        }

        if(is.oes(x$occurrence)){
            ellipsis$x <- ellipsis$x[ellipsis$y!=0];
            ellipsis$y <- ellipsis$y[ellipsis$y!=0];
        }
        # Remove NAs
        if(any(is.na(ellipsis$x))){
            ellipsis$x <- ellipsis$x[!is.na(ellipsis$x)];
            ellipsis$y <- ellipsis$y[!is.na(ellipsis$y)];
        }

        if(!any(names(ellipsis)=="main")){
            if(type=="abs"){
                ellipsis$main <- "|Residuals| vs Fitted";
            }
            else{
                ellipsis$main <- "Residuals^2 vs Fitted";
            }
        }

        if(!any(names(ellipsis)=="xlab")){
            ellipsis$xlab <- "Fitted";
        }
        if(!any(names(ellipsis)=="ylab")){
            if(type=="abs"){
                ellipsis$ylab <- "|Residuals|";
            }
            else{
                ellipsis$ylab <- "Residuals^2";
            }
        }

        do.call(plot,ellipsis);
        abline(h=0, col="grey", lty=2);
        if(lowess){
            lines(lowess(ellipsis$x, ellipsis$y), col="red");
        }
    }

    # 6. Q-Q with the specified distribution
    plot4 <- function(x, ...){
        ellipsis <- list(...);

        ellipsis$y <- residuals(x);
        if(is.oes(x$occurrence)){
            ellipsis$y <- ellipsis$y[actuals(x$occurrence)!=0];
        }

        if(!any(names(ellipsis)=="xlab")){
            ellipsis$xlab <- "Theoretical Quantile";
        }
        if(!any(names(ellipsis)=="ylab")){
            ellipsis$ylab <- "Actual Quantile";
        }

        if(any(x$loss==c("MAEh","TMAE","GTMAE","MACE"))){
            if(!any(names(ellipsis)=="main")){
                ellipsis$main <- "QQ-plot of Laplace distribution";
            }
            ellipsis$x <- qlaplace(ppoints(500), mu=0, scale=x$scale);

            do.call(qqplot, ellipsis);
            qqline(ellipsis$y, distribution=function(p) qlaplace(p, mu=0, scale=x$scale));
        }
        else if(any(x$loss==c("HAMh","THAM","GTHAM","CHAM"))){
            if(!any(names(ellipsis)=="main")){
                ellipsis$main <- "QQ-plot of S distribution";
            }
            ellipsis$x <- qs(ppoints(500), mu=0, scale=x$scale);

            do.call(qqplot, ellipsis);
            qqline(ellipsis$y, distribution=function(p) qs(p, mu=0, scale=x$scale));
        }
        else{
            if(!any(names(ellipsis)=="main")){
                ellipsis$main <- "QQ plot of normal distribution";
            }

            do.call(qqnorm, ellipsis);
            qqline(ellipsis$y);
        }
    }

    # 7 and 8. ACF and PACF
    plot5 <- function(x, type="acf", ...){
        ellipsis <- list(...);

        if(!any(names(ellipsis)=="main")){
            if(type=="acf"){
                ellipsis$main <- "Autocorrelation Function";
            }
            else{
                ellipsis$main <- "Partial Autocorrelation Function";
            }
        }

        if(!any(names(ellipsis)=="xlab")){
            ellipsis$xlab <- "Lags";
        }
        if(!any(names(ellipsis)=="ylab")){
            if(type=="acf"){
                ellipsis$ylab <- "ACF";
            }
            else{
                ellipsis$ylab <- "PACF";
            }
        }

        if(!any(names(ellipsis)=="ylim")){
            ellipsis$ylim <- c(-1,1);
        }

        if(type=="acf"){
            theValues <- acf(residuals(x), plot=FALSE, na.action=na.pass);
        }
        else{
            theValues <- pacf(residuals(x), plot=FALSE, na.action=na.pass);
        }
        ellipsis$x <- theValues$acf[-1];

        ellipsis$type <- "h"

        do.call(plot,ellipsis);
        abline(h=0, col="black", lty=1);
        abline(h=qnorm(c((1-level)/2, (1+level)/2),0,sqrt(1/nobs(x))), col="red", lty=2);
    }

    # 9. Plot of states
    plot6 <- function(x, ...){
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
            ellipsis$actuals <- actuals(x);
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

    # Do plots
    if(any(which==1)){
        plot1(x, ...);
    }

    if(any(which==2)){
        plot2(x, ...);
    }

    if(any(which==3)){
        plot2(x, "rstudent", ...);
    }

    if(any(which==4)){
        plot3(x, ...);
    }

    if(any(which==5)){
        plot3(x, type="squared", ...);
    }

    if(any(which==6)){
        plot4(x, ...);
    }

    if(any(which==7)){
        plot5(x, type="acf", ...);
    }

    if(any(which==8)){
        plot5(x, type="pacf", ...);
    }

    if(any(which==9)){
        plot6(x, ...);
    }
}

#' @export
plot.smoothC <- function(x, ...){
    graphmaker(actuals(x), x$forecast, x$fitted, x$lower, x$upper, x$level,
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
    if(any(x$interval!=c("none","n"))){
        graphmaker(actuals(x$model),x$mean,fitted(x$model),x$lower,x$upper,x$level,main=x$method,...);
    }
    else{
        graphmaker(actuals(x$model),x$mean,fitted(x$model),main=x$method,...);
    }
}

#' @export
plot.oesg <- function(x, ...){
    # This is needed, because OESG models have two pairs of residuals.
    plot.smooth(x, which=1, ...);
}

#### Prints of smooth ####
#' @export
print.smooth <- function(x, ...){
    ellipsis <- list(...);
    if(!any(names(ellipsis)=="digits")){
        digits <- 4;
    }
    else{
        digits <- ellipsis$digits;
    }
    smoothType <- smoothType(x);

    if(!is.list(x$model)){
        if(smoothType=="CMA"){
            holdout <- FALSE;
            interval <- FALSE;
            cumulative <- FALSE;
        }
        else{
            holdout <- any(!is.na(x$holdout));
            interval <- any(!is.na(x$lower));
            cumulative <- x$cumulative;
        }
    }
    else{
        holdout <- any(!is.na(x$holdout));
        interval <- any(!is.na(x$lower));
        cumulative <- x$cumulative;
    }

    if(all(holdout,interval)){
        if(!cumulative){
            insideinterval <- sum((x$holdout <= x$upper) & (x$holdout >= x$lower)) / length(x$forecast) * 100;
        }
        else{
            insideinterval <- NULL;
        }
    }
    else{
        insideinterval <- NULL;
    }

    intervalType <- x$interval;

    if(!is.null(x$model)){
        if(!is.list(x$model)){
            if(any(smoothType==c("SMA","CMA"))){
                x$iprob <- 1;
                x$initialType <- "b";
                occurrence <- "n";
            }
            else if(smoothType=="ETS"){
                # If cumulative forecast and Etype=="M", report that this was "parameteric" interval
                if(cumulative & substr(modelType(x),1,1)=="M"){
                    intervalType <- "p";
                }
            }
        }
    }
    if(is.oes(x$occurrence)){
        occurrence <- x$occurrence$occurrence;
    }
    else{
        occurrence <- "n";
    }

    ssOutput(x$timeElapsed, x$model, persistence=x$persistence, transition=x$transition, measurement=x$measurement,
             phi=x$phi, ARterms=x$AR, MAterms=x$MA, constant=x$constant, A=x$A, B=x$B,initialType=x$initialType,
             nParam=x$nParam, s2=x$s2, hadxreg=!is.null(x$xreg), wentwild=x$updateX,
             loss=x$loss, cfObjective=x$lossValue, interval=interval, cumulative=cumulative,
             intervalType=intervalType, level=x$level, ICs=x$ICs,
             holdout=holdout, insideinterval=insideinterval, errormeasures=x$accuracy,
             occurrence=occurrence, obs=nobs(x), digits=digits);
}

#' @export
print.smooth.sim <- function(x, ...){
    ellipsis <- list(...);
    if(!any(names(ellipsis)=="digits")){
        digits <- 4;
    }
    else{
        digits <- ellipsis$digits;
    }

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
            print(round(xPersistence,digits));
            if(x$phi!=1){
                cat(paste0("Phi: ",x$phi,"\n"));
            }
            if(any(x$occurrence!=1)){
                cat(paste0("The data is produced based on an occurrence model.\n"));
            }
            cat(paste0("True likelihood: ",round(x$logLik,digits),"\n"));
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
                print(round(ARterms,digits));
            }
            if(!is.null(x$MA)){
                cat(paste0("MA parameters: \n"));
                print(round(MAterms,digits));
            }
            if(!is.na(x$constant)){
                cat(paste0("Constant value: ",round(x$constant,digits),"\n"));
            }
            cat(paste0("True likelihood: ",round(x$logLik,digits),"\n"));
        }
        else if(smoothType=="CES"){
            cat(paste0("Smoothing parameter A: ",round(x$A,digits),"\n"));
            if(!is.null(x$B)){
                if(is.complex(x$B)){
                    cat(paste0("Smoothing parameter B: ",round(x$B,digits),"\n"));
                }
                else{
                    cat(paste0("Smoothing parameter b: ",round(x$B,digits),"\n"));
                }
            }
            cat(paste0("True likelihood: ",round(x$logLik,digits),"\n"));
        }
        else if(smoothType=="SMA"){
            cat(paste0("True likelihood: ",round(x$logLik,digits),"\n"));
        }
    }
}

#' @export
print.smooth.forecast <- function(x, ...){
    if(any(x$interval!=c("none","n"))){
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

#' @export
print.oes <- function(x, ...){
    ellipsis <- list(...);
    if(!any(names(ellipsis)=="digits")){
        digits <- 4;
    }
    else{
        digits <- ellipsis$digits;
    }

    occurrence <- x$occurrence
    if(occurrence=="g"){
        occurrence <- "General";
    }
    else if(occurrence=="d"){
        occurrence <- "Direct probability";
    }
    else if(occurrence=="f"){
        occurrence <- "Fixed probability";
    }
    else if(occurrence=="i"){
        occurrence <- "Inverse odds ratio";
    }
    else if(occurrence=="o"){
        occurrence <- "Odds ratio";
    }
    else{
        occurrence <- "None";
    }
    ICs <- round(c(AIC(x),AICc(x),BIC(x),BICc(x)),digits);
    names(ICs) <- c("AIC","AICc","BIC","BICc");
    cat(paste0("Occurrence state space model estimated: ",occurrence,"\n"));
    if(!is.null(x$model)){
        cat(paste0("Underlying ETS model: ",x$model,"\n"));
    }
    if(!is.null(x$persistence)){
        cat("Smoothing parameters:\n");
        print(round(x$persistence[,1],digits));
    }
    if(!is.null(x$initial)){
        cat("Vector of initials:\n");
        print(round(x$initial,digits));
    }
    if(!is.null(sigma(x))){
        cat("\nError standard deviation: "); cat(round(sigma(x),digits));
    }
    cat("\nSample size: "); cat(nobs(x));
    cat("\nNumber of estimated parameters: "); cat(nparam(x));
    cat("\nNumber of degrees of freedom: "); cat(nobs(x)-nparam(x));
    cat("\nInformation criteria: \n");
    print(ICs);
}

#### Residuals for provided object ####
#' @export
residuals.smooth <- function(object, ...){
    if(errorType(object)=="A"){
        return(object$residuals);
    }
    else{
        return(log(1+object$residuals));
    }
}

#' @importFrom stats rstandard
#' @export
rstandard.smooth <- function(model, ...){
    obs <- nobs(model);
    df <- obs - nparam(model);

    # If this is an occurrence model, then only modify the non-zero obs
    if(is.oes(model$occurrence)){
        residsToGo <- which(actuals(model$occurrence)!=0);
    }
    else{
        residsToGo <- c(1:obs);
    }

    errors <- residuals(model);
    return((errors - mean(errors[residsToGo], na.rm=TRUE)) / sqrt(sigma(model)^2 * obs / df));
}

#' @importFrom stats rstudent
#' @export
rstudent.smooth <- function(model, ...){
    obs <- nobs(model);
    df <- obs - nparam(model) - 1;
    rstudentised <- errors <- residuals(model);
    errors[] <- errors - mean(errors, na.rm=TRUE);
    # If this is an occurrence model, then only modify the non-zero obs
    if(is.oes(model$occurrence)){
        residsToGo <- which(actuals(model$occurrence)!=0);
    }
    else{
        residsToGo <- c(1:obs);
    }
    # Prepare the residuals
    if(errorType(model)=="M"){
        for(i in residsToGo){
            rstudentised[i] <- errors[i] / sqrt(sum(errors[-i]^2, na.rm=TRUE) / df);
        }
    }
    else{
        for(i in residsToGo){
            rstudentised[i] <- errors[i] / sqrt(sum(errors[-i]^2, na.rm=TRUE) / df);
        }
    }
    return(rstudentised);
}



#### Simulate data using provided object ####
#' @importFrom utils tail
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

    loss <- object$loss;
    if(any(loss==c("MAE","MAEh","TMAE","GTMAE","MACE"))){
        randomizer <- "rlaplace";
        if(!is.null(ellipsis$mu)){
            args$mu <- ellipsis$mu;
        }
        else{
            args$mu <- 0;
        }

        if(!is.null(ellipsis$scale)){
            args$scale <- ellipsis$scale;
        }
        else{
            args$scale <- mean(abs(residuals(object)));
        }
    }
    else if(any(loss==c("HAM","HAMh","THAM","GTHAM","CHAM"))){
        randomizer <- "rs";
        if(!is.null(ellipsis$mu)){
            args$mu <- ellipsis$mu;
        }
        else{
            args$mu <- 0;
        }

        if(!is.null(ellipsis$scale)){
            args$scale <- ellipsis$scale;
        }
        else{
            args$scale <- mean(sqrt(abs(residuals(object))));
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
    args$frequency <- frequency(actuals(object));
    args$obs <- obs;
    args$nsim <- nsim;
    args$initial <- object$initial;
    # If this is an occurrence model, use the fitted values for the probabilities
    if(is.list(object$occurrence)){
        args$iprob <- fitted(object$occurrence);
    }
    else{
        args$iprob <- 1;
    }

    if(smoothType=="ETS"){
        model <- modelType(object);
        if(any(unlist(gregexpr("C",model))==-1)){
            args$model <- model;
            args$phi <- object$phi;
            args$persistence <- object$persistence;
            args$initialSeason <- object$initialSeason;

            simulatedData <- do.call("sim.es",args);
        }
        else{
            message("Sorry, but we cannot simulate data from combined model.");
            simulatedData <- NA;
        }
    }
    else if(smoothType=="ARIMA"){
        if(any(unlist(gregexpr("combine",object$model))==-1)){
            args$orders <- orders(object);
            args$lags <- lags(object);
            args$AR <- object$AR;
            args$MA <- object$MA;
            args$constant <- object$constant;

            simulatedData <- do.call("sim.ssarima",args);
        }
        else{
            message("Sorry, but we cannot simulate data from combined model.");
            simulatedData <- NA;
        }
    }
    else if(smoothType=="CES"){
        args$seasonality <- modelType(object);
        args$A <- object$A;
        args$B <- object$B;

        simulatedData <- do.call("sim.ces",args);
    }
    else if(smoothType=="GUM"){
        args$orders <- orders(object);
        args$lags <- lags(object);
        args$measurement <- object$measurement;
        args$transition <- object$transition;
        args$persistence <- object$persistence;

        simulatedData <- do.call("sim.gum",args);
    }
    else if(smoothType=="SMA"){
        args$order <- orders(object);

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
