% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/autoces.R
\name{auto.ces}
\alias{auto.ces}
\title{Complex Exponential Smoothing Auto}
\usage{
auto.ces(y, models = c("none", "simple", "full"),
  initial = c("backcasting", "optimal"), ic = c("AICc", "AIC", "BIC",
  "BICc"), loss = c("likelihood", "MSE", "MAE", "HAM", "MSEh", "TMSE",
  "GTMSE", "MSCE"), h = 10, holdout = FALSE, cumulative = FALSE,
  interval = c("none", "parametric", "likelihood", "semiparametric",
  "nonparametric"), level = 0.95, bounds = c("admissible", "none"),
  silent = c("all", "graph", "legend", "output", "none"), xreg = NULL,
  regressors = c("use", "select"), initialX = NULL, ...)
}
\arguments{
\item{y}{Vector or ts object, containing data needed to be forecasted.}

\item{models}{The vector containing several types of seasonality that should
be used in CES selection. See \link[smooth]{ces} for more details about the
possible types of seasonal models.}

\item{initial}{Can be either character or a vector of initial states. If it
is character, then it can be \code{"optimal"}, meaning that the initial
states are optimised, or \code{"backcasting"}, meaning that the initials are
produced using backcasting procedure.}

\item{ic}{The information criterion used in the model selection procedure.}

\item{loss}{The type of Loss Function used in optimization. \code{loss} can
be: \code{likelihood} (assuming Normal distribution of error term),
\code{MSE} (Mean Squared Error), \code{MAE} (Mean Absolute Error),
\code{HAM} (Half Absolute Moment), \code{TMSE} - Trace Mean Squared Error,
\code{GTMSE} - Geometric Trace Mean Squared Error, \code{MSEh} - optimisation
using only h-steps ahead error, \code{MSCE} - Mean Squared Cumulative Error.
If \code{loss!="MSE"}, then likelihood and model selection is done based
on equivalent \code{MSE}. Model selection in this cases becomes not optimal.

There are also available analytical approximations for multistep functions:
\code{aMSEh}, \code{aTMSE} and \code{aGTMSE}. These can be useful in cases
of small samples.

Finally, just for fun the absolute and half analogues of multistep estimators
are available: \code{MAEh}, \code{TMAE}, \code{GTMAE}, \code{MACE}, \code{TMAE},
\code{HAMh}, \code{THAM}, \code{GTHAM}, \code{CHAM}.}

\item{h}{Length of forecasting horizon.}

\item{holdout}{If \code{TRUE}, holdout sample of size \code{h} is taken from
the end of the data.}

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

\item{bounds}{What type of bounds to use in the model estimation. The first
letter can be used instead of the whole word.}

\item{silent}{If \code{silent="none"}, then nothing is silent, everything is
printed out and drawn. \code{silent="all"} means that nothing is produced or
drawn (except for warnings). In case of \code{silent="graph"}, no graph is
produced. If \code{silent="legend"}, then legend of the graph is skipped.
And finally \code{silent="output"} means that nothing is printed out in the
console, but the graph is produced. \code{silent} also accepts \code{TRUE}
and \code{FALSE}. In this case \code{silent=TRUE} is equivalent to
\code{silent="all"}, while \code{silent=FALSE} is equivalent to
\code{silent="none"}. The parameter also accepts first letter of words ("n",
"a", "g", "l", "o").}

\item{xreg}{The vector (either numeric or time series) or the matrix (or
data.frame) of exogenous variables that should be included in the model. If
matrix included than columns should contain variables and rows - observations.
Note that \code{xreg} should have number of observations equal either to
in-sample or to the whole series. If the number of observations in
\code{xreg} is equal to in-sample, then values for the holdout sample are
produced using \link[smooth]{es} function.}

\item{regressors}{The variable defines what to do with the provided xreg:
\code{"use"} means that all of the data should be used, while
\code{"select"} means that a selection using \code{ic} should be done.
\code{"combine"} will be available at some point in future...}

\item{initialX}{The vector of initial parameters for exogenous variables.
Ignored if \code{xreg} is NULL.}

\item{...}{Other non-documented parameters.  For example \code{FI=TRUE}
will make the function produce Fisher Information matrix, which then can be
used to calculated variances of parameters of the model.}
}
\value{
Object of class "smooth" is returned. See \link[smooth]{ces} for
details.
}
\description{
Function estimates CES in state space form with information potential equal
to errors with different seasonality types and chooses the one with the
lowest IC value.
}
\details{
The function estimates several Complex Exponential Smoothing in the
state space 2 described in Svetunkov, Kourentzes (2015) with the information
potential equal to the approximation error using different types of
seasonality and chooses the one with the lowest value of information
criterion.

For some more information about the model and its implementation, see the
vignette: \code{vignette("ces","smooth")}
}
\examples{

y <- ts(rnorm(100,10,3),frequency=12)
# CES with and without holdout
auto.ces(y,h=20,holdout=TRUE)
auto.ces(y,h=20,holdout=FALSE)


# Selection between "none" and "full" seasonalities
\donttest{auto.ces(AirPassengers,h=8,holdout=TRUE,
                   models=c("n","f"),interval="p",level=0.8,ic="AIC")}

ourModel <- auto.ces(AirPassengers,interval="sp")

summary(ourModel)
forecast(ourModel)
plot(forecast(ourModel))

}
\references{
\itemize{
\item Svetunkov, I., Kourentzes, N. (February 2015). Complex exponential
smoothing. Working Paper of Department of Management Science, Lancaster
University 2015:1, 1-31.
\item Svetunkov I., Kourentzes N. (2017) Complex Exponential Smoothing
for Time Series Forecasting. Not yet published.
}
}
\seealso{
\code{\link[smooth]{ces}, \link[smooth]{es},
\link[greybox]{forecast}, \link[stats]{ts}}
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
