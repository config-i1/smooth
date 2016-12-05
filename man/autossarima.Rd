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
auto.ssarima(data, orders=list(ar=c(3,3),i=c(2,1),ma=c(3,3)), lags=c(1,frequency(data)),
        combine=FALSE, workFast=TRUE,
        initial=c("backcasting","optimal"), ic=c("AICc","AIC","BIC"),
        cfType=c("MSE","MAE","HAM","MLSTFE","MSTFE","MSEh"),
        h=10, holdout=FALSE,
        intervals=c("none","parametric","semiparametric","nonparametric"), level=0.95,
        intermittent=c("none","auto","fixed","croston","tsb","sba"),
        bounds=c("admissible","none"),
        silent=c("none","all","graph","legend","output"),
        xreg=NULL, updateX=FALSE, ...)
}
%- maybe also 'usage' for other objects documented here.
\arguments{

  \item{data}{
    Data that needs to be forecasted.
  }
  \item{orders}{
    List of maximum orders to check, containing vector variables \code{ar}, \code{i} and \code{ma}. If a variable is not provided in the list, then it is assumed to be equal to zero. At least one variable should have the same length as \code{lags}.
  }
  \item{lags}{
    Defines lags for the corresponding orders (see examples). The length of \code{lags} must correspond to the length of either \code{ar.orders} or \code{i.orders} or \code{ma.orders}. There is no restrictions on the length of \code{lags} vector.
  }
  \item{combine}{
    If \code{TRUE}, then resulting ARIMA is combined using AIC weights.
  }
  \item{workFast}{
    If \code{TRUE}, then some of the orders of ARIMA are skipped. This is not advised for models with \code{lags} greater than 12.
  }
  \item{initial}{
    Character value which defines how the model is initialised: it can be \code{"optimal"}, meaning that the initial states are optimised, or \code{"backcasting"}, meaning that the initials are produced using backcasting procedure.
  }
  \item{ic}{
    Information criterion to use in model selection.
  }
  \item{cfType}{
    Type of Cost Function used in optimization. \code{cfType} can be: \code{MSE} (Mean Squared Error), \code{MAE} (Mean Absolute Error), \code{HAM} (Half Absolute Moment), \code{MLSTFE} - Mean Log Squared Trace Forecast Error, \code{MSTFE} - Mean Squared Trace Forecast Error and \code{MSEh} - optimisation using only h-steps ahead error. If \code{cfType!="MSE"}, then likelihood and model selection is done based on equivalent \code{MSE}. Model selection in this cases becomes not optimal.

    There are also available analytical approximations for multistep functions: \code{aMSEh}, \code{aMSTFE} and \code{aMLSTFE}. These can be useful in cases of small samples.
  }
  \item{h}{
    The forecasting horizon.
  }
  \item{holdout}{
    If \code{TRUE}, the holdout sample of size h will be taken from the data. If \code{FALSE}, no holdout is defined.
  }
  \item{intervals}{
    Type of intervals to construct. This can be:

    \itemize{
        \item \code{none}, aka \code{n} - do not produce prediction intervals.

        \item \code{parametric}, \code{p} - use state-space structure of ETS. In case of mixed models this is done using simulations, which may take longer time than for the pure additive and pure multiplicative models.

        \item \code{semiparametric}, \code{sp} - intervals based on covariance matrix of 1 to h steps ahead errors and assumption of normal / log-normal distribution (depending on error type).

        \item \code{nonparametric}, \code{np} - intervals based on values from a quantile regression on error matrix (see Taylor and Bunn, 1999). The model used in this process is e[j] = a j^b, where j={1,..,h}.

        %\item Finally \code{asymmetric} are based on half moment of distribution.
    }

    The parameter also accepts \code{TRUE} and \code{FALSE}. Former means that parametric intervals are constructed, while latter is equivalent to \code{none}.
  }
  \item{level}{
    Confidence level. Defines width of prediction interval.
  }
  \item{intermittent}{
    Defines type of intermittent model used. Can be: 1. \code{none}, meaning that the data should be considered as non-intermittent; 2. \code{fixed}, taking into account constant Bernoulli distribution of demand occurancies; 3. \code{croston}, based on Croston, 1972 method with SBA correction; 4. \code{tsb}, based on Teunter et al., 2011 method. 5. \code{auto} - automatic selection of intermittency type based on information criteria. The first letter can be used instead. 6. \code{"sba"} - Syntetos-Boylan Approximation for Croston's method (bias correction) discussed in Syntetos and Boylan, 2005.
  }
  \item{bounds}{
    What type of bounds to use for the smoothing parameters. The first letter can be used instead of the whole word.
  }
  \item{silent}{
    If \code{silent="none"}, then nothing is silent, everything is printed out and drawn. \code{silent="all"} means that nothing is produced or drawn (except for warnings). In case of \code{silent="graph"}, no graph is produced. If \code{silent="legend"}, then legend of the graph is skipped. And finally \code{silent="output"} means that nothing is printed out in the console, but the graph is produced. \code{silent} also accepts \code{TRUE} and \code{FALSE}. In this case \code{silent=TRUE} is equivalent to \code{silent="all"}, while \code{silent=FALSE} is equivalent to \code{silent="none"}. The parameter also accepts first letter of words ("n", "a", "g", "l", "o").
  }
  \item{xreg}{
    Vector (either numeric or time series) or matrix (or data.frame) of exogenous variables that should be included in the model. If matrix included than columns should contain variables and rows - observations. Note that \code{xreg} should have number of observations equal either to in-sample or to the whole series. If the number of observations in \code{xreg} is equal to in-sample, then values for the holdout sample are produced using Naive.
  }
  \item{updateX}{
    If \code{TRUE}, transition matrix for exogenous variables is estimated, introducing non-linear interractions between parameters. Prerequisite - non-NULL \code{xreg}.
  }
  \item{...}{
    Other non-documented parameters. For example \code{FI=TRUE} will make the function also produce Fisher Information matrix, which then can be used to calculated variances of parameters of the model.
    Maximum orders to check can also be specified separately, however \code{orders} variable must be set to \code{NULL}:
    \code{ar.orders} - Maximum order of AR term. Can be vector, defining max orders of AR, SAR etc.
    \code{i.orders} - Maximum order of I. Can be vector, defining max orders of I, SI etc.
    \code{ma.orders} - Maximum order of MA term. Can be vector, defining max orders of MA, SMA etc.
  }
}
\details{
    The function constructs bunch of ARIMAs in Single Source of Error State-space form (see \link[smooth]{ssarima} documentation) and selects the best one based on information criterion.

    Due to the flexibility of the model, multiple seasonalities can be used. For example, something crazy like this can be constructed: SARIMA(1,1,1)(0,1,1)[24](2,0,1)[24*7](0,0,1)[24*30], but the estimation may take a lot of time...
}
\value{
  Object of class "smooth" is returned. See \link[smooth]{ssarima} for details.
}
\references{
    \enumerate{
        \item{Snyder, R. D., 1985. Recursive Estimation of Dynamic Linear Models. Journal of the Royal Statistical Society, Series B (Methodological) 47 (2), 272-276.}
        \item{Hyndman, R.J., Koehler, A.B., Ord, J.K., and Snyder, R.D. (2008) Forecasting with exponential smoothing: the state space approach, Springer-Verlag. \url{http://www.exponentialsmoothing.net}.}
        \item{Teunter R., Syntetos A., Babai Z. (2011). Intermittent demand: Linking forecasting to inventory obsolescence. European Journal of Operational Research, 214, 606-615.}
        \item{Croston, J. (1972) Forecasting and stock control for intermittent demands. Operational Research Quarterly, 23(3), 289-303.}
        \item{Syntetos, A., Boylan J. (2005) The accuracy of intermittent demand estimates. International Journal of Forecasting, 21(2), 303-314.}
    }
}
\author{
    Ivan Svetunkov
}

\seealso{
    \code{\link[forecast]{ets}, \link[smooth]{es}, \link[smooth]{ces}, \link[smooth]{sim.es}, \link[smooth]{ges}, \link[smooth]{ssarima}}
}

\examples{
x <- rnorm(118,100,3)

# The best ARIMA for the data
ourModel <- auto.ssarima(x,orders=list(ar=c(2,1),i=c(1,1),ma=c(2,1)),lags=c(1,12),
                     h=18,holdout=TRUE,intervals="np")

# The other one using optimised states
\dontrun{auto.ssarima(x,orders=list(ar=c(3,2),i=c(2,1),ma=c(3,2)),lags=c(1,12),
                     initial="o",h=18,holdout=TRUE)}

# And now combined ARIMA
\dontrun{auto.ssarima(x,orders=list(ar=c(3,2),i=c(2,1),ma=c(3,2)),lags=c(1,12),
                      combine=TRUE,h=18,holdout=TRUE)}

summary(ourModel)
forecast(ourModel)
plot(forecast(ourModel))

}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{ SARIMA }
\keyword{ ARIMA }% __ONLY ONE__ keyword per line
