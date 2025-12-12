#' Sparse ARMA Model in State Space Form
#'
#' @description
#' Implements a Sparse ARMA model in the State Space form.
#' Unlike standard ARIMA which expands polynomials,
#' this function directly maps AR and MA orders to specific lags.
#'
#' @param data Vector or ts object with the data
#' @param orders Vector c(p,q) specifying AR and MA orders (default: c(1,1))
#' @param constant Logical, whether to include a constant term (default: FALSE)
#' @param loss Loss function type.
#' @param h Forecast horizon (default: 0)
#' @param holdout Logical, whether to use holdout sample (default: FALSE)
#' @param arma List with ar and ma parameters if they do not need to be estimated
#' @param initial Initialisation method for states
#' @param bounds Parameter bounds
#' @param silent Logical, whether to suppress output (default: TRUE)
#'
#' @return Object of class c("adam", "smooth") containing:
#' \itemize{
#'   \item model - Model name
#'   \item timeElapsed - Computation time
#'   \item data - Input data
#'   \item holdout - Holdout sample (if applicable)
#'   \item fitted - Fitted values
#'   \item residuals - Residuals
#'   \item forecast - Point forecasts if h>0
#'   \item states - State matrix
#'   \item persistence - Persistence vector (g)
#'   \item transition - Transition matrix (F)
#'   \item measurement - Measurement matrix (W)
#'   \item B - Vector of estimated parameters
#'   \item orders - Orders specified by the user
#'   \item constant - Constant value (if included)
#'   \item arma - vector of ARMA parameters
#'   \item initial - Initial state values
#'   \item initialType - Type of initialisation
#'   \item nParam - Number of parameters
#'   \item logLik - Log-likelihood value
#'   \item loss - Loss function used in the estimation
#'   \item lossValue - Value of the loss function
#'   \item accuracy - Accuracy measures
#' }
#'
#' @details
#' The model implements: y_t = phi * y_{t-p} + theta * epsilon_{t-q} + epsilon_t
#'
#' State Space Form:
#' - Measurement equation: y_t = w' * v_{t-l} + epsilon_t
#' - Transition equation: v_t = F * v_{t-l} + g * epsilon_t
#'
#' @examples
#' \dontrun{
#' # Fit SpARMA(1,1) model
#' model <- sparma(AirPassengers, orders=c(1,1), h=12, holdout=TRUE)
#'
#' # Fit SpARMA(2,3) with constant
#' model <- sparma(rnorm(100), orders=c(2,3), constant=TRUE)
#'
#' # Provide fixed parameters
#' model <- sparma(rnorm(100), orders=c(1,1), arma=c(0.7,0.5))
#' }
#'
#' export
sparma <- function(data, orders=c(1,1), constant=FALSE,
                   loss=c("likelihood","MSE","MAE","HAM","LASSO","RIDGE","MSEh","TMSE","GTMSE","MSCE"),
                   h=0, holdout=FALSE, arma=NULL,
                   initial=c("backcasting","optimal","two-stage","complete"),
                   bounds=c("none","usual","admissible"), silent=TRUE, ...) {

    # Start timer
    startTime <- Sys.time()

    # ===== ARGUMENT VALIDATION =====
    loss <- match.arg(loss)
    initial <- match.arg(initial)
    bounds <- match.arg(bounds)

    ellipsis <- list(...);

    # Validate orders
    if(length(orders) != 2) {
        stop("orders must be a vector of length 2: c(p,q)")
    }
    p <- orders[1]
    q <- orders[2]

    if(p < 0 || q < 0) {
        stop("Orders must be non-negative")
    }

    if(p == 0 && q == 0 && !constant) {
        stop("At least one of p or q must be greater than 0")
    }

    # State space dimension
    K <- max(p, q)
    lags <- 1:K

    # Convert orders to list format
    orders_list <- list(ar=p, i=0, ma=q)


    # Build model string for parametersChecker
    model <- "NNN"
    yName <- deparse(substitute(data))

    # Default parameters for parametersChecker
    outliers <- NULL
    level <- 0.95
    persistence <- NULL
    phi <- NULL
    distribution <- "dnorm"
    occurrence <- "none"
    ic <- "AICc"
    regressors <- "use"
    formula <- NULL
    modelDo <- ""

    # Call parametersChecker
    checkerReturn <- smooth:::parametersChecker(
        data=data, model=model, lags=lags, formula=formula,
        orders=orders_list, constant=constant, arma=NULL,
        outliers=outliers, level=level,
        persistence=persistence, phi=phi, initial=initial,
        distribution=distribution, loss=loss, h=h, holdout=holdout,
        occurrence=occurrence, ic=ic, bounds=bounds,
        regressors=regressors, yName=yName,
        silent=silent, modelDo=modelDo,
        ParentEnvironment=environment(), ellipsis=ellipsis, fast=FALSE
    )

    #### Hack the outputs of the function to align with sparma ####

    if(obsInSample <= K + 1) {
        stop("Not enough observations for the specified orders")
    }

    # Handle arma parameter input
    if(!is.null(arma)) {
        if(length(arma)==2){
            armaParameters <- arma;

            arEstimate <- maEstimate <- FALSE;
        }
        else{
            warning("arma needs to be of length 2. I'll ignore it and estimate the parameters.")

            arEstimate <- maEstimate <- TRUE;
        }
    }

    lagsModelARIMA <- lagsModelAll <- matrix(sort(unique(c(p,q))), ncol=1);
    lagsModelMax <- max(lagsModelAll);
    initialArimaNumber <- componentsNumberARIMA <- length(lagsModelAll);
    componentsNamesARIMA <- componentsNamesARIMA[lagsModelAll];
    refineHead <- TRUE

    # These two are not used and can be ignored
    # print(nonZeroARI)
    # print(nonZeroMA)

    # Initial parameter values
    if(arRequired && arEstimate) {
        ARValue <- 0.1
    } else if(arRequired) {
        ARValue <- armaParameters[1]
    } else {
        ARValue <- 0
    }

    if(maRequired && maEstimate) {
        MAValue <- 0.1
    } else if(maRequired) {
        MAValue <- armaParameters[2]
    } else {
        MAValue <- 0
    }

    if(constantRequired) {
        constantValue <- mean(yInSample)
        lagsModelAll <- matrix(c(lagsModelAll,1), ncol=1);
    } else {
        constantValue <- 0
    }


    # Create state space matrices
    matricesCreated <- sparmaMatricesCreator(K, p, q, armaParameters,
                                             arRequired, arEstimate,
                                             maRequired, maEstimate,
                                             obsInSample,
                                             lagsModellAll, lagsModelMax,
                                             componentsNumberARIMA,
                                             componentsNamesARIMA,
                                             constantRequired, constantName)

    matVt <- matricesCreated$matVt
    matWt <- matricesCreated$matWt
    matF <- matricesCreated$matF
    vecG <- matricesCreated$vecG

    # Create profiles for C++ fitter
    lagsModelAll <- lags
    profilesList <- adamProfilesCreator(lagsModelAll)
    indexLookupTable <- profilesList$indexLookupTable
    profilesRecentTable <- profilesList$profilesRecent

    # Cost function using C++ fitter
    CF <- function(B) {
        # Fill matrices with parameters from B
        matricesFilled <- sparmaMatricesFiller(B, matricesCreated,
                                               arEstimate, maEstimate, constantEstimate,
                                               arRequired, maRequired, constantRequired,
                                               p, q, K, initialType, bounds)

        # Fit using C++ function
        adamFitted <- smooth:::adamFitterWrap(
            matricesFilled$matVt, matricesFilled$matWt, matricesFilled$matF, matricesFilled$vecG,
            lagsModelAll, indexLookupTable, profilesRecentTable,
            Etype, Ttype, Stype, componentsNumberETS, componentsNumberETSSeasonal,
            componentsNumberARIMA, xregNumber, constantRequired,
            yInSample, ot, any(initialType==c("complete","backcasting")),
            nIterations, refineHead, adamETS
        )

        errors <- adamFitted$errors
        yFitted <- adamFitted$yFitted

        # Calculate loss
        CFValue <- switch(loss,
                          "likelihood" = {
                              n <- length(errors)
                              sigma2 <- sum(errors^2) / n
                              if(sigma2 <= 0) sigma2 <- 1e-10
                              n * log(2 * pi * sigma2) / 2 + n / 2
                          },
                          "MSE" = mean(errors^2),
                          "MAE" = mean(abs(errors)),
                          "HAM" = mean(sqrt(abs(errors))),
                          "LASSO" = {
                              lambda <- 0.01
                              mean(errors^2) + lambda * (abs(matricesFilled$ARValue) + abs(matricesFilled$MAValue))
                          },
                          "RIDGE" = {
                              lambda <- 0.01
                              mean(errors^2) + lambda * (matricesFilled$ARValue^2 + matricesFilled$MAValue^2)
                          },
                          "MSEh" = {
                              if(length(errors) > h && h > 0) {
                                  mean(errors[(length(errors)-h+1):length(errors)]^2)
                              } else {
                                  mean(errors^2)
                              }
                          },
                          "TMSE" = mean(errors^2),
                          "GTMSE" = mean(errors^2),
                          "MSCE" = mean(cumsum(errors)^2),
                          mean(errors^2)
        )

        if(!is.finite(CFValue) || is.na(CFValue)) {
            CFValue <- 1e100
        }

        return(CFValue)
    }

    # Build initial parameter vector
    B0 <- numeric(0)

    if(arEstimate && arRequired) {
        B0 <- c(B0, atanh(ARValue * 0.5))
    }
    if(maEstimate && maRequired) {
        B0 <- c(B0, atanh(MAValue * 0.5))
    }
    if(constantEstimate) {
        B0 <- c(B0, constantValue)
    }
    if(initialType == "optimal") {
        B0 <- c(B0, matVt[1,])
    }

    # Optimize
    if(length(B0) > 0) {
        opt <- nloptr(
            x0 = B0,
            eval_f = CF,
            opts = list(
                algorithm = "NLOPT_LN_SBPLX",
                maxeval = 1000,
                xtol_rel = 1e-6,
                ftol_rel = 1e-8
            )
        )

        B <- opt$solution
    } else {
        B <- B0
    }

    # Count parameters
    parametersNumber[1,1] <- parametersNumber[1,1] + length(B);


    # Final fit with optimized parameters
    matricesFinal <- sparmaMatricesFiller(B, matricesCreated,
                                          arEstimate, maEstimate, constantEstimate,
                                          arRequired, maRequired, constantRequired,
                                          p, q, K, initialType, bounds)

    adamFittedFinal <- smooth:::adamFitterWrap(
        matricesFinal$matVt, matricesFinal$matWt, matricesFinal$matF, matricesFinal$vecG,
        lagsModelAll, indexLookupTable, profilesRecentTable,
        Etype, Ttype, Stype, componentsNumberETS, componentsNumberETSSeasonal,
        componentsNumberARIMA, xregNumber, constantRequired,
        yInSample, ot, any(initialType==c("complete","backcasting")),
        nIterations, refineHead, adamETS
    )

    errors <- adamFittedFinal$errors
    yFitted <- adamFittedFinal$yFitted
    matVt <- adamFittedFinal$matVt

    # Calculate final loss and logLik
    n <- obsInSample
    sigma2 <- sum(errors^2) / n
    if(sigma2 <= 0) sigma2 <- 1e-10

    lossValue <- switch(loss,
                        "likelihood" = n * log(2 * pi * sigma2) / 2 + n / 2,
                        "MSE" = mean(errors^2),
                        "MAE" = mean(abs(errors)),
                        mean(errors^2))

    logLik <- -n * log(2 * pi * sigma2) / 2 - n / 2

    # Forecasting if h > 0
    if(h > 0) {
        adamForecasted <- smooth:::adamForecasterWrap(
            matricesFinal$matVt, matricesFinal$matF, matricesFinal$matWt, matricesFinal$vecG,
            lagsModelAll, h, Etype, Ttype, Stype,
            componentsNumberETS, componentsNumberETSSeasonal, componentsNumberARIMA,
            xregNumber, constantRequired, obsInSample, ot
        )

        yForecast <- adamForecasted$yForecast
        matVtForecast <- adamForecasted$matVt
    } else {
        yForecast <- NULL
        matVtForecast <- NULL
    }

    # Calculate accuracy using greybox::measures
    if(holdout && h > 0) {
        errorsHoldout <- yHoldout - yForecast
        accuracyMeasures <- greybox::measures(yHoldout, yForecast, yInSample)
    } else {
        accuracyMeasures <- greybox::measures(yInSample, yFitted, yInSample)
    }

    # Build model name
    modelName <- paste0("SpARMA(", p, ",", q, ")")
    if(constantRequired) modelName <- paste0(modelName, " with constant")

    # Elapsed time
    timeElapsed <- as.numeric(difftime(Sys.time(), startTime, units="secs"))

    # Build output object
    output <- list(
        model = modelName,
        timeElapsed = timeElapsed,
        data = dataOriginal,
        holdout = yHoldout,
        fitted = yFitted,
        residuals = errors,
        forecast = yForecast,
        states = matVt,
        persistence = matricesFinal$vecG,
        transition = matricesFinal$matF,
        measurement = matricesFinal$matWt[1,],
        lagVector = lagsModelAll,
        orders = orders,
        constant = if(constantRequired) matricesFinal$constantValue else NULL,
        AR = if(arRequired) matricesFinal$ARValue else NULL,
        MA = if(maRequired) matricesFinal$MAValue else NULL,
        initial = matricesFinal$matVt[1,],
        initialType = initialType,
        nParam = nParam,
        logLik = logLik,
        lossValue = lossValue,
        lossFunction = loss,
        accuracy = accuracyMeasures,
        bounds = bounds,
        y = yInSample,
        obs = obsInSample,
        obsAll = obsAll,
        h = h,
        holdout_used = holdout,
        call = match.call()
    )

    class(output) <- c("adam", "smooth")

    # Print if not silent
    if(!silent && !is.null(yForecast)) {
        plot(output, which=7)
    }

    return(output)
}


# Helper function: Create initial state space matrices
sparmaMatricesCreator <- function(K, p, q, armaParameters,
                                  arRequired, arEstimate,
                                  maRequired, maEstimate,
                                  obsInSample,
                                  lagsModellAll, lagsModelMax,
                                  componentsNumberARIMA,
                                  componentsNamesARIMA,
                                  constantRequired, constantName) {

    # Build measurement matrix (rows = observations, cols = states)
    matWt <- matrix(1, nrow=obsInSample, ncol=componentsNumberARIMA+constantRequired,
                    dimnames=list(NULL, c(componentsNamesARIMA, constantName)));

    vecG <- matrix(0, componentsNumberARIMA+constantRequired, 1);

    # Build transition matrix F
    matF <- matrix(0, componentsNumberARIMA+constantRequired, componentsNumberARIMA+constantRequired);

    # Fill in the transition where the AR is present
    if(arRequired && !arEstimate){
        matF[lagsModellAll==p,] <- armaParameters[1];
        vecG[lagsModellAll==p,] <- vecG[lagsModellAll==p,] + armaParameters[1];
    }

    # Fill in the transition where the AR is present
    if(maRequired && !maEstimate){
        vecG[lagsModellAll==q,] <- vecG[lagsModellAll==q,] + armaParameters[2];
    }

    if(constantRequired){
        matF[componentsNumberARIMA+constantRequired, componentsNumberARIMA+constantRequired] <- 1;
    }

    # Initialize state matrix
    matVt <- matrix(0, componentsNumberARIMA+constantRequired, obsInSample+lagsModelMax,
                    dimnames=list(c(componentsNamesARIMA, constantName), NULL))

    return(list(matVt = matVt, matWt = matWt, matF = matF, vecG = vecG));
}


# Helper function: Fill matrices with parameters from vector B
sparmaMatricesFiller <- function(B, matricesCreated,
                                 arEstimate, maEstimate, constantEstimate,
                                 arRequired, maRequired, constantRequired,
                                 p, q, K, initialType, bounds) {

    idx <- 1

    # Extract AR parameter
    if(arEstimate && arRequired) {
        ARVal <- B[idx]
        idx <- idx + 1

        # Apply bounds
        if(bounds == "admissible") {
            ARVal <- tanh(ARVal) * 0.99
        } else if(bounds == "usual") {
            ARVal <- tanh(ARVal)
        }
    } else {
        ARVal <- matricesCreated$ARValue
    }

    # Extract MA parameter
    if(maEstimate && maRequired) {
        MAVal <- B[idx]
        idx <- idx + 1

        # Apply bounds
        if(bounds == "admissible") {
            MAVal <- tanh(MAVal) * 0.99
        } else if(bounds == "usual") {
            MAVal <- tanh(MAVal)
        }
    } else {
        MAVal <- matricesCreated$MAValue
    }

    # Extract constant
    if(constantEstimate) {
        constVal <- B[idx]
        idx <- idx + 1
    } else {
        constVal <- matricesCreated$constantValue
    }

    # Extract initial states
    matVt <- matricesCreated$matVt
    if(initialType == "optimal" && length(B) >= idx + K - 1) {
        matVt[1,] <- B[idx:(idx+K-1)]
    }

    # Update eta and psi vectors
    eta <- rep(0, K)
    if(p > 0 && p <= K) {
        eta[p] <- ARVal
    }

    psi <- rep(0, K)
    if(q > 0 && q <= K) {
        psi[q] <- MAVal
    }

    # Update persistence vector
    vecG <- eta + psi

    # Update transition matrix
    matF <- matrix(0, nrow=K, ncol=K)
    for(i in 1:K) {
        matF[i,] <- rep(eta[i], K)
    }

    # Measurement matrix (unchanged structure)
    matWt <- matricesCreated$matWt

    return(list(
        matVt = matVt,
        matWt = matWt,
        matF = matF,
        vecG = vecG,
        ARValue = ARVal,
        MAValue = MAVal,
        constantValue = constVal
    ))
}

#'
#'
#' #' @export
#' print.sparma <- function(x, ...) {
#'   cat("\nSparse ARMA Model\n")
#'   cat("=================\n")
#'   cat("Model:", x$model, "\n")
#'   cat("Time elapsed:", round(x$timeElapsed, 4), "seconds\n")
#'   cat("Initial type:", x$initialType, "\n")
#'   cat("\nOrders: AR(", x$orders[1], "), MA(", x$orders[2], ")\n", sep="")
#'
#'   if(!is.null(x$AR)) {
#'     cat("AR parameter:", round(x$AR, 4), "\n")
#'   }
#'   if(!is.null(x$MA)) {
#'     cat("MA parameter:", round(x$MA, 4), "\n")
#'   }
#'   if(!is.null(x$constant)) {
#'     cat("Constant:", round(x$constant, 4), "\n")
#'   }
#'
#'   cat("\nSample size:", x$obs, "\n")
#'   cat("Number of parameters:", x$nParam, "\n")
#'   cat("Loss function:", x$lossFunction, "\n")
#'   cat("Loss value:", round(x$lossValue, 4), "\n")
#'   cat("Log-likelihood:", round(x$logLik, 4), "\n")
#'
#'   cat("\nAccuracy measures:\n")
#'   print(round(x$accuracy, 4))
#'
#'   invisible(x)
#' }
#'
#'
#' #' @export
#' summary.sparma <- function(object, ...) {
#'   cat("\nSparse ARMA Model - Summary\n")
#'   cat("===========================\n")
#'   print(object)
#'
#'   cat("\n\nState Space Components:\n")
#'   cat("Measurement vector (w):\n")
#'   print(object$measurement)
#'   cat("\nPersistence vector (g):\n")
#'   print(object$persistence)
#'   cat("\nTransition matrix (F):\n")
#'   print(object$transition)
#'
#'   cat("\nInitial states:\n")
#'   print(object$initial)
#'
#'   if(object$h > 0 && !is.null(object$forecast)) {
#'     cat("\nForecast (", object$h, " steps ahead):\n", sep="")
#'     print(round(object$forecast, 4))
#'   }
#'
#'   invisible(object)
#' }
#'
#'
#' #' @export
#' plot.sparma <- function(x, which=7, ...) {
#'
#'   if(which == 7 || which == "forecast") {
#'     # Plot fit and forecast
#'     if(x$holdout_used) {
#'       allY <- c(x$y, x$holdout)
#'       allFitted <- c(x$fitted, x$forecast)
#'
#'       plot(allY, type="l", col="black", lwd=2,
#'            main=paste(x$model, "\nFitted and Forecast"),
#'            ylab="Value", xlab="Time", ...)
#'       lines(x$fitted, col="blue", lwd=1.5)
#'       lines((length(x$y)+1):length(allY), x$forecast, col="red", lwd=1.5)
#'       abline(v=length(x$y), lty=2, col="gray")
#'       legend("topleft", legend=c("Actual", "Fitted", "Forecast", "Holdout start"),
#'              col=c("black", "blue", "red", "gray"), lty=c(1,1,1,2),
#'              lwd=c(2,1.5,1.5,1))
#'     } else {
#'       plot(x$y, type="l", col="black", lwd=2,
#'            main=paste(x$model, "\nFitted and Forecast"),
#'            ylab="Value", xlab="Time", ...)
#'       lines(x$fitted, col="blue", lwd=1.5)
#'       if(x$h > 0) {
#'         lines((length(x$y)+1):(length(x$y)+x$h), x$forecast,
#'               col="red", lwd=1.5)
#'         legend("topleft", legend=c("Actual", "Fitted", "Forecast"),
#'                col=c("black", "blue", "red"), lty=1, lwd=c(2,1.5,1.5))
#'       } else {
#'         legend("topleft", legend=c("Actual", "Fitted"),
#'                col=c("black", "blue"), lty=1, lwd=c(2,1.5))
#'       }
#'     }
#'   } else if(which == 1 || which == "residuals") {
#'     # Residual plot
#'     plot(x$residuals, type="l", main="Residuals",
#'          ylab="Residuals", xlab="Time", ...)
#'     abline(h=0, col="red", lty=2)
#'   } else if(which == 2 || which == "fitted") {
#'     # Fitted vs Actual
#'     plot(x$y, x$fitted, main="Fitted vs Actual",
#'          xlab="Actual", ylab="Fitted", ...)
#'     abline(a=0, b=1, col="red", lty=2)
#'   } else if(which == 3 || which == "states") {
#'     # States plot
#'     matplot(x$states, type="l", main="State Variables",
#'             ylab="State Value", xlab="Time", ...)
#'     legend("topleft", legend=paste("State", 1:ncol(x$states)),
#'            col=1:ncol(x$states), lty=1:ncol(x$states))
#'   } else if(which == 4 || which == "acf") {
#'     # ACF of residuals
#'     acf(x$residuals, main="ACF of Residuals", ...)
#'   } else if(which == 5 || which == "pacf") {
#'     # PACF of residuals
#'     pacf(x$residuals, main="PACF of Residuals", ...)
#'   } else if(which == 6 || which == "qq") {
#'     # Q-Q plot
#'     qqnorm(x$residuals, main="Normal Q-Q Plot", ...)
#'     qqline(x$residuals, col="red")
#'   }
#'
#'   invisible(x)
#' }
#'
#'
#' #' @export
#' forecast.sparma <- function(object, h=NULL, ...) {
#'   if(is.null(h)) {
#'     h <- object$h
#'   }
#'
#'   if(h == 0 || h == object$h) {
#'     return(object$forecast)
#'   }
#'
#'   # Re-forecast with new horizon
#'   nStates <- length(object$initial)
#'   matVt <- rbind(object$states, matrix(0, nrow=1, ncol=nStates))
#'   matVt[nrow(matVt),] <- object$states[nrow(object$states),]
#'
#'   yForecast <- numeric(h)
#'   constVal <- ifelse(is.null(object$constant), 0, object$constant)
#'
#'   for(i in 1:h) {
#'     t <- nrow(matVt)
#'     yForecast[i] <- sum(object$measurement * matVt[t,]) + constVal
#'     matVt <- rbind(matVt,
#'                    matrix(object$transition %*% matVt[t,], nrow=1))
#'   }
#'
#'   return(yForecast)
#' }
#'
#'
#' #' @export
#' coef.sparma <- function(object, ...) {
#'   coefs <- c()
#'
#'   if(!is.null(object$AR)) {
#'     coefs <- c(coefs, AR=object$AR)
#'   }
#'   if(!is.null(object$MA)) {
#'     coefs <- c(coefs, MA=object$MA)
#'   }
#'   if(!is.null(object$constant)) {
#'     coefs <- c(coefs, constant=object$constant)
#'   }
#'
#'   return(coefs)
#' }
#'
#'
#' #' @export
#' residuals.sparma <- function(object, ...) {
#'   return(object$residuals)
#' }
#'
#'
#' #' @export
#' fitted.sparma <- function(object, ...) {
#'   return(object$fitted)
#' }
