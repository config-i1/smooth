\name{auto.ssarima}
\alias{auto.ssarima}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
State-Space ARIMA
}
\description{
Function selects the best State-Space ARIMA based on information criteria, using fancy branch and bound mechanism. The resulting model can be not optimal in IC meaning, but it is usually reasonable.
}
\usage{
auto.ssarima(data, ar.max=c(3), i.max=c(2), ma.max=c(3), lags=c(1),
             initial=c("backcasting","optimal"), IC=c("AICc","AIC","BIC"),
             CF.type=c("MSE","MAE","HAM","MLSTFE","TFL","MSTFE","MSEh"),
             h=10, holdout=FALSE, intervals=FALSE, int.w=0.95,
             int.type=c("parametric","semiparametric","nonparametric","asymmetric"),
             silent=FALSE, legend=TRUE,
             xreg=NULL, go.wild=FALSE)
}
%- maybe also 'usage' for other objects documented here.
\arguments{

  \item{data}{
    The data that needs to be forecasted.
  }
  \item{ar.max}{
    Maximum order of AR term. Can be vector, defining max orders of AR, SAR etc.
  }
  \item{i.max}{
    Maximum order of I. Can be vector, defining max orders of I, SI etc.
  }
  \item{ma.max}{
    Maximum order of MA term. Can be vector, defining max orders of MA, SMA etc.
  }
  \item{lags}{
    Defines lags for the corresponding orders (see examples above). The length of \code{lags} must correspond to the length of either \code{ar.orders} or \code{i.orders} or \code{ma.orders}. There is no restrictions on the length of \code{lags} vector.
  }
  \item{initial}{
    Can be either character or a vector of initial states. If it is character, then it can be \code{"optimal"}, meaning that the initial states are optimised, or \code{"backcasting"}, meaning that the initials are produced using backcasting procedure.
  }
  \item{IC}{
    The information criterion to use in the model selection.
  }
  \item{CF.type}{
    Type of Cost Function used in optimization. \code{CF.type} can be: \code{MSE} (Mean Squared Error), \code{MAE} (Mean Absolute Error), \code{HAM} (Half Absolute Moment), \code{TFL} - Trace forecast likelihood, \code{MLSTFE} - Mean Log Squared Trace Forecast Error, \code{MSTFE} - Mean Squared Trace Forecast Error and \code{MSEh} - optimisation using only h-steps ahead error. If \code{CF.type!="MSE"} or \code{CF.type!="TFL"}, then the likelihood and model selection is done based the equivalent \code{MSE} or \code{TFL} respectively.
  }
  \item{h}{
    The forecasting horizon.
  }
  \item{holdout}{
    If \code{TRUE}, the holdout of the size \code{h} is taken from the end of the data.
  }
  \item{intervals}{
    If \code{TRUE}, the prediction interval is constructed.
  }
  \item{int.w}{
    Defines the width of the prediction interval.
  }
  \item{int.type}{
    The type of intervals to construct. The first letter can be used instead of the whole word. \code{parametric} use the state-space structure of ETS, \code{semiparametric} are based on the covariance matrix of 1 to h steps ahead errors and normal distribution, \code{nonparametric} use the quantiles of the errors in that matrix, \code{asymmetric} are based on half moment of distribution.
  }
  \item{silent}{
    If \code{TRUE}, no output or graph is given.
  }
  \item{legend}{
    If \code{TRUE}, the legend is drawn on the output plot.
  }
  \item{xreg}{
    The vector (either numeric or time series) or matrix (or data.frame) of exogenous variables that should be included in the model. If matrix included than columns should contain variables and rows - observations. Note that \code{xreg} should have the number of observations equal either to in-sample or to the whole series. If the number of observations in \code{xreg} is equal to in-sample, then the values in the holdout sample are forecasted using Naive.
  }
  \item{go.wild}{
    If \code{TRUE}, the transition matrix for exogenous variables is estimated, introducing non-linear interractions between the parameters. Prerequisite - non-NULL \code{xreg}.
  }
}
\details{
The function constructs ARIMA in the Single Source of Error State-space form (see \link[smooth]{ssarima} documentation).

Due to the flexibility of the model, multiple seasonalities can be used. For example, something crazy like this can be constructed: SARIMA(1,1,1)(0,1,1)[24](2,0,1)[24*7](0,0,1)[24*30], but the estimation may take a lot of time...
}
\value{
  The list of the values from \link[smooth]{ssarima} is return.
}
\references{
    \enumerate{
        \item{Hyndman, R.J., Koehler, A.B., Ord, J.K., and Snyder, R.D. (2008) Forecasting with exponential smoothing: the state space approach, Springer-Verlag. \url{http://www.exponentialsmoothing.net}.}
    }
}
\author{
    Ivan Svetunkov
}

\seealso{
    \code{\link[forecast]{ets}, \link[smooth]{es}, \link[smooth]{ces}, \link[smooth]{sim.ets}, \link[smooth]{ges}, \link[smooth]{ssarima}}
}

\examples{
x <- rnorm(118,100,3)

# The best ARIMA for the data
auto.ssarima(x,ar.max=c(3,2),i.max=c(2,1),ma.max=c(3,2),lags=c(1,12),
             h=18,holdout=TRUE)->test

# The other one using optimised states
auto.ssarima(x,ar.max=c(3,2),i.max=c(2,1),ma.max=c(3,2),lags=c(1,12),
             initial="o",h=18,holdout=TRUE)->test

}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{ SARIMA }
\keyword{ ARIMA }% __ONLY ONE__ keyword per line
