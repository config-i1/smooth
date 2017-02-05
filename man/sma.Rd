\name{sma}
\alias{sma}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
Simple Moving Average
}
\description{
    Function constructs State-Space simple moving average of predefined order
}
\usage{
sma(data, order=NULL, ic=c("AICc","AIC","BIC"),
    h=10, holdout=FALSE,
    intervals=c("none","parametric","semiparametric","nonparametric"), level=0.95,
    silent=c("none","all","graph","legend","output"), ...)
}
%- maybe also 'usage' for other objects documented here.
\arguments{

  \item{data}{
    Data that needs to be forecasted.
  }
  \item{order}{
    Order of simple moving average. If \code{NULL}, then it is selected automatically using information criteria.
  }
  \item{ic}{
    Information criterion to use in order selection.
  }
  \item{h}{
    Length of forecasting horizon.
  }
  \item{holdout}{
    If \code{TRUE}, holdout sample of size \code{h} is taken from the end of the data.
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
  \item{silent}{
    If \code{silent="none"}, then nothing is silent, everything is printed out and drawn. \code{silent="all"} means that nothing is produced or drawn (except for warnings). In case of \code{silent="graph"}, no graph is produced. If \code{silent="legend"}, then legend of the graph is skipped. And finally \code{silent="output"} means that nothing is printed out in the console, but the graph is produced. \code{silent} also accepts \code{TRUE} and \code{FALSE}. In this case \code{silent=TRUE} is equivalent to \code{silent="all"}, while \code{silent=FALSE} is equivalent to \code{silent="none"}. The parameter also accepts first letter of words ("n", "a", "g", "l", "o").
  }
  \item{...}{
    Other non-documented parameters.
    For example parameter \code{model} can accept a previously estimated SMA model and use its parameters.
  }
}
\details{
The function constructs ARIMA in the Single Source of Error State-space form (first proposed in Snyder, 1985):

\eqn{y_[t] = w' v_[t-l] + \epsilon_[t]}

\eqn{v_[t] = F v_[t-1] + g \epsilon_[t]}

Where \eqn{v_[t]} is a state vector (defined using \code{order}).

}
\value{
  Object of class "smooth" is returned. It contains the list of the following values:

  \itemize{
    \item \code{model} - the name of the estimated model.
    \item \code{timeElapsed} - time elapsed for the construction of the model.
    \item \code{states} - the matrix of the fuzzy components of ssarima, where \code{rows} correspond to time and \code{cols} to states.
    \item \code{transition} - matrix F.
    \item \code{persistence} - the persistence vector. This is the place, where smoothing parameters live.
    \item \code{order} - order of moving average.
    \item \code{initialType} - Typetof initial values used.
    \item \code{nParam} - number of estimated parameters.
    \item \code{fitted} - the fitted values of ETS.
    \item \code{forecast} - the point forecast of ETS.
    \item \code{lower} - the lower bound of prediction interval. When \code{intervals=FALSE} then NA is returned.
    \item \code{upper} - the higher bound of prediction interval. When \code{intervals=FALSE} then NA is returned.
    \item \code{residuals} - the residuals of the estimated model.
    \item \code{errors} - The matrix of 1 to h steps ahead errors.
    \item \code{s2} - variance of the residuals (taking degrees of freedom into account).
    \item \code{intervals} - type of intervals asked by user.
    \item \code{level} - confidence level for intervals.
    \item \code{actuals} - the original data.
    \item \code{holdout} - the holdout part of the original data.
    \item \code{ICs} - values of information criteria of the model. Includes AIC, AICc and BIC.
    \item \code{logLik} - log-likelihood of the function.
    \item \code{cf} - Cost function value.
    \item \code{cfType} - Type of cost function used in the estimation.
    \item \code{accuracy} - vector of accuracy measures for the holdout sample. In case of non-intermittent data includes: MPE, MAPE, SMAPE, MASE, sMAE, RelMAE, sMSE and Bias coefficient (based on complex numbers). In case of intermittent data the set of errors will be: sMSE, sPIS, sCE (scaled cumulative error) and Bias coefficient. This is available only when \code{holdout=TRUE}.
  }
}
\references{
    \enumerate{
        \item{Snyder, R. D., 1985. Recursive Estimation of Dynamic Linear Models. Journal of the Royal Statistical Society, Series B (Methodological) 47 (2), 272-276.}
        \item{Hyndman, R.J., Koehler, A.B., Ord, J.K., and Snyder, R.D. (2008) Forecasting with exponential smoothing: the state space approach, Springer-Verlag. \url{http://www.exponentialsmoothing.net}.}
    }
}
\author{
    Ivan Svetunkov
}

\seealso{
\code{\link[forecast]{ma}, \link[smooth]{es}, \link[smooth]{ssarima}}
}

\examples{
# SMA of specific order
ourModel <- sma(rnorm(118,100,3),order=12,h=18,holdout=TRUE,intervals="p")

# SMA of arbitrary order
ourModel <- sma(rnorm(118,100,3),h=18,holdout=TRUE,intervals="sp")

summary(ourModel)
forecast(ourModel)
plot(forecast(ourModel))
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{ SARIMA }
\keyword{ ARIMA }% __ONLY ONE__ keyword per line
