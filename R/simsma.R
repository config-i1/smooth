#' Simulate Simple Moving Average
#'
#' Function generates data using SMA in a Single Source of Error state space
#' model as a data generating process.
#'
#' For the information about the function, see the vignette:
#' \code{vignette("simulate","smooth")}
#'
#' @template ssSimParam
#' @template ssAuthor
#' @template ssKeywords
#'
#' @template ssGeneralRef
#'
#' @param order Order of the modelled series. If omitted, then a random order from 1 to 100 is selected.
#' @param initial Vector of initial states for the model. If \code{NULL},
#' values are generated.
#' @param ...  Additional parameters passed to the chosen randomizer. All the
#' parameters should be passed in the order they are used in chosen randomizer.
#' For example, passing just \code{sd=0.5} to \code{rnorm} function will lead
#' to the call \code{rnorm(obs, mean=0.5, sd=1)}.
#'
#' @return List of the following values is returned:
#' \itemize{
#' \item \code{model} - Name of SMA model.
#' \item \code{data} - Time series vector (or matrix if \code{nsim>1}) of the generated
#' series.
#' \item \code{states} - Matrix (or array if \code{nsim>1}) of states. States are in
#' columns, time is in rows.
#' \item \code{initial} - Vector (or matrix) of initial values.
#' \item \code{probability} - vector of probabilities used in the simulation.
#' \item \code{intermittent} - type of the intermittent model used.
#' \item \code{residuals} - Error terms used in the simulation. Either vector or matrix,
#' depending on \code{nsim}.
#' \item \code{occurrence} - Values of occurrence variable. Once again, can be either
#' a vector or a matrix...
#' \item \code{logLik} - Log-likelihood of the constructed model.
#' }
#'
#' @seealso \code{\link[smooth]{es}, \link[forecast]{ets},
#' \link[forecast]{forecast}, \link[stats]{ts}, \link[stats]{Distributions}}
#'
#' @examples
#'
#' # Create 40 observations of quarterly data using AAA model with errors from normal distribution
#' sma10 <- sim.sma(order=10,frequency=4,obs=40,randomizer="rnorm",mean=0,sd=100)
#'
#' @export sim.sma
sim.sma <- function(order=NULL, obs=10, nsim=1,
                   frequency=1,
                   initial=NULL,
                   randomizer=c("rnorm","rt","rlaplace","rs"),
                   probability=1, ...){
    # Function generates data using SMA model as a data generating process.
    #    Copyright (C) 2017 Ivan Svetunkov

    randomizer <- randomizer[1];

    if(is.null(order)){
        order <- ceiling(runif(1,0,100));
    }

    # In the case of wrong nsim, make it natural number. The same is for obs and frequency.
    nsim <- abs(round(nsim,0));
    obs <- abs(round(obs,0));
    frequency <- abs(round(frequency,0));

    # Check the inital vector length
    if(!is.null(initial)){
        if(order!=length(initial)){
            warning(paste0("The length of initial state vector does not correspond to the chosen model!\n",
                           "Falling back to random number generator."),call.=FALSE);
            initial <- NULL;
        }
    }

    ARIMAModel <- sim.ssarima(orders=list(ar=order,i=0,ma=0), lags=1,
                              obs=obs, nsim=nsim,
                              frequency=frequency, AR=rep(1/order,order), MA=NULL, constant=FALSE,
                              initial=initial, bounds="none",
                              randomizer=randomizer,
                              probability=probability, ...)

    ARIMAModel$model <- paste0("SMA(",order,")");
    if(any(probability!=1)){
        ARIMAModel$model <- paste0("i",ARIMAModel$model);
    }
    ARIMAModel$AR <- NULL;
    ARIMAModel$MA <- NULL;
    ARIMAModel$constant <- NULL;
    return(ARIMAModel)
}
