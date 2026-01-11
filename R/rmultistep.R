#' Multiple steps ahead forecast errors
#'
#' The function extracts 1 to h steps ahead forecast errors from the model.
#'
#' The errors correspond to the error term epsilon_t in the ETS models. Don't forget
#' that different models make different assumptions about epsilon_t and / or 1+epsilon_t.
#'
#' @template ssAuthor
#' @template ssKeywords
#'
#' @param object Model estimated using one of the forecasting functions.
#' @param h The forecasting horizon to use.
#' @param error Defines what type of error to return. \code{"default"} means returning the
#' one used in the original model. \code{"additive"} is to return e_t = y_t - mu_t.
#' Finally, \code{"multiplicative"} will return e_t = (y_t - mu_t) / mu_t.
#' @param ... Currently nothing is accepted via ellipsis.
#' @return The matrix with observations in rows and h steps ahead values in columns.
#' So, the first row corresponds to the forecast produced from the 0th observation
#' from 1 to h steps ahead.
#' @seealso \link[stats]{residuals},
#' @examples
#'
#' x <- rnorm(100,0,1)
#' ourModel <- adam(x)
#' rmultistep(ourModel, h=13)
#'
#' @export rmultistep
rmultistep <- function(object, h=10,
                       error=c("default","additive","multiplicative"),
                       ...) UseMethod("rmultistep")

#' @export
rmultistep.default <- function(object, h=10,
                               error=c("default","additive","multiplicative"),
                               ...){
    return(NULL);
}

#' @export
rmultistep.adam <- function(object, h=10,
                            error=c("default","additive","multiplicative"),
                            ...){
    error <- match.arg(error);

    yClasses <- class(actuals(object));

    adamETS <- adamETSChecker(object);

    # Model type
    model <- modelType(object);
    Etype <- switch(error,
                    "additive"="A",
                    "multiplicative"="M",
                    "default"=errorType(object));
    Ttype <- trendType(object);
    Stype <- seasonType(object);

    # Technical parameters
    lagsModelAll <- modelLags(object);
    lagsModelMax <- max(lagsModelAll);
    lagsOriginal <- lags(object);
    if(Ttype!="N"){
        lagsOriginal <- c(1,lagsOriginal);
    }

    # Get componentsNumberETS, seasonal and componentsNumberARIMA
    componentsDefined <- componentsDefiner(object);
    componentsNumberETS <- componentsDefined$componentsNumberETS;
    componentsNumberETSSeasonal <- componentsDefined$componentsNumberETSSeasonal;
    componentsNumberETSNonSeasonal <- componentsDefined$componentsNumberETSNonSeasonal;
    componentsNumberARIMA <- componentsDefined$componentsNumberARIMA;
    constantRequired <- componentsDefined$constantRequired;

    if(ncol(object$data)>1){
        xregNumber <- ncol(object$data)-1;
    }
    else{
        xregNumber <- 0;
    }
    obsInSample <- nobs(object);

    # Create C++ adam class, which will then use fit, forecast etc methods
    adamCpp <- new(adamCore,
                   lagsModelAll, Etype, Ttype, Stype,
                   componentsNumberETSNonSeasonal,
                   componentsNumberETSSeasonal,
                   componentsNumberETS, componentsNumberARIMA,
                   xregNumber, length(lagsModelAll),
                   constantRequired, adamETS);

    # Function returns the matrix with multi-step errors
    if(is.occurrence(object$occurrence)){
        ot <- matrix(actuals(object$occurrence),obsInSample,1);
    }
    else{
        ot <- matrix(1,obsInSample,1);
    }
    adamProfiles <- adamProfileCreator(lagsModelAll, lagsModelMax, obsInSample,
                                       lagsOriginal, time(actuals(object)), yClasses);
    # profilesRecentTable <- adamProfiles$recent;
    indexLookupTable <- adamProfiles$lookup;

    # Fill in the profile. This is done in Errorer as well, but this is just in case
    profilesRecentTable <- object$profileInitial;

    # Return multi-step errors matrix
    if(any(yClasses=="ts")){
        return(ts(
            adamCpp$ferrors(t(object$states), object$measurement,
                            object$transition,
                            indexLookupTable, profilesRecentTable,
                            h, matrix(actuals(object),obsInSample,1))$errors,
            start=start(actuals(object)), frequency=frequency(actuals(object))));
    }
    else{
        return(zoo(
            adamCpp$ferrors(t(object$states), object$measurement,
                            object$transition,
                            indexLookupTable, profilesRecentTable,
                            h, matrix(actuals(object),obsInSample,1))$errors,
            order.by=time(actuals(object))));
    }
}
