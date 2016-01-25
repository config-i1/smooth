\name{ges}
\alias{ges}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
General Exponential Smoothing
}
\description{
Function builds General Exponential Smoothing, estimating matrices F, w, vector g and initial parameters.
}
\usage{
ges(data, bounds=FALSE, seasonality=c("N","S","F"),
    trace=FALSE, CF.type=c("TLV","GV","TV","hsteps","MAD","HAM"),
    use.test=FALSE, intervals=FALSE, int.w=0.95,
    int.type=c("parametric","semiparametric","nonparametric"),
    xreg=NULL, holdout=FALSE, h=10, silent=FALSE, legend=TRUE,
    ...)
}
%- maybe also 'usage' for other objects documented here.
\arguments{

  \item{data}{ the data that needs to be forecasted.
  }
  \item{bounds}{ if \code{TRUE}, the stability condition is taken into account during the optimization.
  }
  \item{seasonality}{ Can be \code{N} - None, \code{S} - Simple (just seasonal lags of states) and \code{F} - Full (combination of previous two). NOT WORKING CURRENTLY!
  }
  \item{trace}{ if \code{TRUE}, trace likelihood is used in the estimation
  }
  \item{CF.type}{ type of trace likelihood cost function: \code{TV} - Total Variation, \code{GV} - Generalized Variance (not working properly yet), \code{TLV} - Total Logarithmic Variance, \code{hsteps} - optimisation using only h-steps ahead error.
  }
  \item{use.test}{
    If \code{TRUE} the statistical test is used on all the found parameters to define if they are significantly different from 1 and 0. If they are not then they are fixed to have the corresponding values. If both tests are not declined, nothing is changed dut to the uncertainty.
  }
  \item{intervals}{ If \code{TRUE}, the prediction interval is constructed.
  }
  \item{int.w}{ Defines the width of the prediction interval.
  }
  \item{int.type}{ The type of intervals to construct. The first letter can be used instead of the whole word. \code{parametric} use the state-space structure of ETS, \code{semiparametric} are based on the covariance matrix of 1 to h steps ahead errors and normal distribution, \code{nonparametric} use the quantiles of the errors in that matrix.
  }
  \item{holdout}{ if \code{TRUE}, the holdout of the size \code{h} is taken from the end of the data.
  }
  \item{xreg}{ The vector / matrix / dataframe of exogenous variables.
  }
  \item{h}{ the forecasting horizon.
  }
  \item{silent}{ if \code{TRUE}, no output or graph is given.
  }
  \item{legend}{ if \code{TRUE}, the legend is drawn on the output plot.
  }
  \item{...}{ All sorts of additional parameters. NOT IMPLEMENTED YET!
  }
}
\details{
The function creates the model of the following type:
\eqn{y_[t] = w' x_[t-1] + \epsilon_[t]}
\eqn{x_[t] = F x_[t-1] + g \epsilon_[t]}

}
\value{
  The list of the following values is returned:

  \itemize{
    \item \code{persistence} - the persistence vector.
    \item \code{mat.F} - matrix F.
    \item \code{mat.w} - matrix w.
    \item \code{states} - the matrix of the components of CES. The included minimum is "level" and "potential". In the case of seasonal model the seasonal component is also included. In the case of exogenous variables the estimated coefficients for the exogenous variables are also included.
    \item \code{fitted} - the fitted values of ETS.
    \item \code{forecast} - the point forecast of ETS.
    \item \code{lower} - the lower bound of prediction interval. When \code{intervals=FALSE} then NA is returned.
    \item \code{upper} - the higher bound of prediction interval. When \code{intervals=FALSE} then NA is returned.
    \item \code{residuals} - the residuals of the estimated model.
    \item \code{actuals} - the original data.
    \item \code{holdout} - the holdout part of the original data.
    \item \code{ICs} - values of information criteria of the model. Includes AIC, AICc and BIC.
    \item \code{CF} - Cost function value.
    \item \code{FI} - Fisher Information estimated in the optimal point.
    \item \code{xreg} - the provided vector or matrix of exogenous variables.
    \item \code{accuracy} - the vector or accuracy measures. Includes MPE, MAPE, SMAPE, MASE and MASALE. Available only when \code{holdout=T}.
  }
}
\references{
No references yet.
}
\author{
Ivan Svetunkov
}

\seealso{
\code{\link[forecast]{ets}, \link[prognosis]{es}, \link[prognosis]{sim.ets}}
}

\examples{

ges(rnorm(118,100,3),h=18,holdout=TRUE,bounds=FALSE,trace=TRUE)->test

ges(rnorm(118,100,3),h=18,holdout=TRUE,bounds=TRUE)->test

}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{ ges }
\keyword{ Exponential Smoothing }% __ONLY ONE__ keyword per line