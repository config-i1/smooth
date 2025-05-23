% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/adam-sma.R, R/sma.R
\name{sma}
\alias{sma}
\alias{sma_old}
\title{Simple Moving Average}
\usage{
sma(y, order = NULL, ic = c("AICc", "AIC", "BIC", "BICc"), h = 10,
  holdout = FALSE, silent = TRUE, fast = TRUE, ...)

sma_old(y, order = NULL, ic = c("AICc", "AIC", "BIC", "BICc"), h = 10,
  holdout = FALSE, cumulative = FALSE, interval = c("none", "parametric",
  "likelihood", "semiparametric", "nonparametric"), level = 0.95,
  silent = c("all", "graph", "legend", "output", "none"), ...)
}
\arguments{
\item{y}{Vector or ts object, containing data needed to be forecasted.}

\item{order}{Order of simple moving average. If \code{NULL}, then it is
selected automatically using information criteria.}

\item{ic}{The information criterion used in the model selection procedure.}

\item{h}{Length of forecasting horizon.}

\item{holdout}{If \code{TRUE}, holdout sample of size \code{h} is taken from
the end of the data.}

\item{silent}{accepts \code{TRUE} and \code{FALSE}. If FALSE, the function
will print its progress and produce a plot at the end.}

\item{fast}{if \code{TRUE}, then the modified Ternary search is used to
find the optimal order of the model. This does not guarantee the optimal
solution, but gives a reasonable one (local minimum).}

\item{...}{Other non-documented parameters.  For example parameter
\code{model} can accept a previously estimated SMA model and use its
parameters.}

\item{cumulative}{If \code{TRUE}, then the cumulative forecast and prediction
interval are produced instead of the normal ones. This is useful for
inventory control systems.}

\item{interval}{Type of interval to construct. This can be:

\itemize{
\item \code{"none"}, aka \code{"n"} - do not produce prediction
interval.
\item \code{"parametric"}, \code{"p"} - use state-space structure of ETS. In
case of mixed models this is done using simulations, which may take longer
time than for the pure additive and pure multiplicative models. This type
of interval relies on unbiased estimate of in-sample error variance, which
divides the sume of squared errors by T-k rather than just T.
\item \code{"likelihood"}, \code{"l"} - these are the same as \code{"p"}, but
relies on the biased estimate of variance from the likelihood (division by
T, not by T-k).
\item \code{"semiparametric"}, \code{"sp"} - interval based on covariance
matrix of 1 to h steps ahead errors and assumption of normal / log-normal
distribution (depending on error type).
\item \code{"nonparametric"}, \code{"np"} - interval based on values from a
quantile regression on error matrix (see Taylor and Bunn, 1999). The model
used in this process is e[j] = a j^b, where j=1,..,h.
}
The parameter also accepts \code{TRUE} and \code{FALSE}. The former means that
parametric interval are constructed, while the latter is equivalent to
\code{none}.
If the forecasts of the models were combined, then the interval are combined
quantile-wise (Lichtendahl et al., 2013).}

\item{level}{Confidence level. Defines width of prediction interval.}
}
\value{
Object of class "smooth" is returned. It contains the list of the
following values:

\itemize{
\item \code{model} - the name of the estimated model.
\item \code{timeElapsed} - time elapsed for the construction of the model.
\item \code{states} - the matrix of the fuzzy components of ssarima, where
\code{rows} correspond to time and \code{cols} to states.
\item \code{transition} - matrix F.
\item \code{persistence} - the persistence vector. This is the place, where
smoothing parameters live.
\item \code{measurement} - measurement vector of the model.
\item \code{order} - order of moving average.
\item \code{initial} - Initial state vector values.
\item \code{initialType} - Type of initial values used.
\item \code{nParam} - table with the number of estimated / provided parameters.
If a previous model was reused, then its initials are reused and the number of
provided parameters will take this into account.
\item \code{fitted} - the fitted values.
\item \code{forecast} - the point forecast.
\item \code{lower} - the lower bound of prediction interval. When
\code{interval=FALSE} then NA is returned.
\item \code{upper} - the higher bound of prediction interval. When
\code{interval=FALSE} then NA is returned.
\item \code{residuals} - the residuals of the estimated model.
\item \code{errors} - The matrix of 1 to h steps ahead errors. Only returned when the
multistep losses are used and semiparametric interval is needed.
\item \code{s2} - variance of the residuals (taking degrees of freedom into
account).
\item \code{interval} - type of interval asked by user.
\item \code{level} - confidence level for interval.
\item \code{cumulative} - whether the produced forecast was cumulative or not.
\item \code{y} - the original data.
\item \code{holdout} - the holdout part of the original data.
\item \code{ICs} - values of information criteria of the model. Includes AIC,
AICc, BIC and BICc.
\item \code{logLik} - log-likelihood of the function.
\item \code{lossValue} - Cost function value.
\item \code{loss} - Type of loss function used in the estimation.
\item \code{accuracy} - vector of accuracy measures for the
holdout sample. Includes: MPE, MAPE, SMAPE, MASE, sMAE, RelMAE, sMSE and
Bias coefficient (based on complex numbers). This is available only when
\code{holdout=TRUE}.
}
}
\description{
Function constructs state space simple moving average of predefined order
}
\details{
The function constructs AR model in the Single Source of Error state space form
based on the idea that:

\eqn{y_{t} = \frac{1}{n} \sum_{j=1}^n y_{t-j}}

which is AR(n) process, that can be modelled using:

\eqn{y_{t} = w' v_{t-1} + \epsilon_{t}}

\eqn{v_{t} = F v_{t-1} + g \epsilon_{t}}

Where \eqn{v_{t}} is a state vector.

For some more information about the model and its implementation, see the
vignette: \code{vignette("sma","smooth")}
}
\examples{

# SMA of specific order
ourModel <- sma(rnorm(118,100,3), order=12, h=18, holdout=TRUE)

# SMA of arbitrary order
ourModel <- sma(rnorm(118,100,3), h=18, holdout=TRUE)

plot(forecast(ourModel, h=18, interval="empirical"))

}
\references{
\itemize{
\item Svetunkov I. (2023) Smooth forecasting with the smooth package in R. arXiv:2301.01790.
\doi{10.48550/arXiv.2301.01790}.
\item Svetunkov I. (2015 - Inf) "smooth" package for R - series of posts about the underlying
models and how to use them: \url{https://openforecast.org/category/r-en/smooth/}.
}

\itemize{
\item Svetunkov, I., & Petropoulos, F. (2017). Old dog, new tricks: a
modelling view of simple moving averages. International Journal of
Production Research, 7543(January), 1-14.
\doi{10.1080/00207543.2017.1380326}
}
}
\seealso{
\code{\link[stats]{filter}, \link[smooth]{adam}, \link[smooth]{msarima}}
}
\author{
Ivan Svetunkov, \email{ivan@svetunkov.com}
}
\keyword{models}
\keyword{nonlinear}
\keyword{regression}
\keyword{smooth}
\keyword{ts}
\keyword{univar}
