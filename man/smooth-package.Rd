\name{smooth}
\alias{smooth}
\docType{package}
\title{
Smooth package
}
\description{
Package contains several exponential (and not) smoothing functions used in time series analysis and forecasting.
}
\details{
\tabular{ll}{
Package: \tab smooth\cr
Type: \tab Package\cr
Version: \tab 0.5\cr
Date: \tab 2016-01-27 - 2016-03-07\cr
License: \tab GPL-2 \cr
}
The following functions are included in the package:
  \itemize{
    \item \link[smooth]{es} - exponential smoothing in Single Source of Errors State Space form.
    \item \link[smooth]{ces} - Complex Exponential Smoothing.
    \item \link[smooth]{ces.auto} - Automatic selection between seasonal and non-seasonal CES.
    \item \link[smooth]{ges} - Generalised Exponential Smoothing.
    \item \link[smooth]{ssarima} - SARIMA in state-space framework.
    \item \link[smooth]{nus} - Non-Uniform Smoothing.
    \item \link[smooth]{sim.ces} - simulate time series using CES as a model.
    \item \link[smooth]{sim.ets} - simulate time series using ETS as a model.
  }
}
\author{
Ivan Svetunkov

Maintainer: Ivan Svetunkov <ivan@svetunkov.ru>
}
\references{
    \itemize{
        \item{Hyndman, R.J., Koehler, A.B., Ord, J.K., and Snyder, R.D. (2008) Forecasting with exponential smoothing: the state space approach, Springer-Verlag. \url{http://www.exponentialsmoothing.net}.}
        \item{Kolassa, S. (2011) Combining exponential smoothing forecasts using Akaike weights. International Journal of Forecasting, 27, pp 238 - 251.}
        \item{Svetunkov, I., Kourentzes, N. (February 2015). Complex exponential smoothing. Working Paper of Department of Management Science, Lancaster University 2015:1, 1-31.}
        \item{Svetunkov I., Kourentzes N. (2016) Complex Exponential Smoothing for Time Series Forecasting. Not yet published.}
        \item{Svetunkov I., Kourentzes N. (2016) Trace forecast likelihood and shrinkage in time series models. Not yet published.}
        \item{Svetunkov S. (2012) Complex-Valued Modeling in Economics and Finance. SpringerLink: Bucher. Springer.}
        \item{Taylor, J.W. and Bunn, D.W. (1999) A Quantile Regression Approach to Generating Prediction Intervals. Management Science, Vol 45, No 2, pp 225-237.}
    }
}
\keyword{ exponential smoothing, ETS, CES, GES, ARIMA, SARIMA, NUS, forecasting, time series}
\seealso{
\code{\link[forecast:forecast]{forecast}}
}
\examples{
y <- rnorm(100,10,3)
y <- ts(y,frequency=12)
y_ces_holdout <- ces.auto(y,h=20,holdout=TRUE)
y_es_holdout <- es(y,h=20,holdout=TRUE)
y_ges_holdout <- ges(y,h=20,holdout=TRUE)
}
