\name{ges}
\alias{ges}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
General Exponential Smoothing
}
\description{
Function constructs General Exponential Smoothing, estimating matrices F, w, vector g and initial parameters.
}
\usage{
ges(data, bounds=TRUE, order=c(2), lags=c(1), initial=NULL,
    persistence=NULL, transition=NULL, measurement=NULL,
    CF.type=c("MSE","MAE","HAM","TLV","GV","TV","hsteps"),
    backcast=FALSE, FI=FALSE, intervals=FALSE, int.w=0.95,
    int.type=c("parametric","semiparametric","nonparametric"),
    xreg=NULL, holdout=FALSE, h=10, silent=FALSE, legend=TRUE,
    ...)
}
%- maybe also 'usage' for other objects documented here.
\arguments{

  \item{data}{ the data that needs to be forecasted.
  }
  \item{bounds}{ if \code{TRUE}, the stability condition is taken into account during the optimization. May take more time.
  }
  \item{order}{ Order of the model. Specified as vector of number of states with different lags. For example, \code{order=c(1,1)} means that there are two states: one of the first lag type, the second of the second type.
  }
  \item{lags}{ Defines lags for the corresponding order. If, for example, \code{order=c(1,1)} and lags defined as \code{lags=c(1,12)}, then the model will have two states, the first of wich has lag 1 and the second has lag 12. The length of  \code{lags} must correspond to the length of \code{order}.
  }
  \item{initial}{ Vector of initial states \eqn{v_[t]} for GES. Should contain \code{order \%*\% lags} elements. If \code{NULL}, then estimated.
  }
  \item{persistence}{ Persistence vector \eqn{g}, containing smoothing parameters. If \code{NULL}, then estimated.
  }
  \item{transition}{ Transition matrix \eqn{F}. Can be provided as a vector. Matrix will be formed using the default \code{matrix(transition,nc,nc)}, where \code{nc} is the number of components in state vector. If \code{NULL}, then estimated.
  }
  \item{measurement}{ Measurement vector \eqn{w}. If \code{NULL}, then estimated.
  }
  \item{CF.type}{ type of trace likelihood cost function: \code{TV} - Total Variation, \code{GV} - Generalized Variance (not working properly yet), \code{TLV} - Total Logarithmic Variance, \code{hsteps} - optimisation using only h-steps ahead error.
  }
  \item{backcast}{ if \code{TRUE}, then the initial state after optimisation is smoothed out using backcast technique. Slows down the optimisation.
  }
  \item{FI}{ If \code{TRUE}, Fisher Information is calculated and produced in the output.
  }
  \item{intervals}{ If \code{TRUE}, the prediction interval is constructed.
  }
  \item{int.w}{ Defines the width of the prediction interval.
  }
  \item{int.type}{ The type of intervals to construct. The first letter can be used instead of the whole word. \code{parametric} use the state-space structure of ETS, \code{semiparametric} are based on the covariance matrix of 1 to h steps ahead errors and normal distribution, \code{nonparametric} use the quantiles of the errors in that matrix.
  }
  \item{xreg}{ The vector / matrix / dataframe of exogenous variables.
  }
  \item{holdout}{ if \code{TRUE}, the holdout of the size \code{h} is taken from the end of the data.
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
The function creates the Single Source of Error State-space model of the following type:

\eqn{y_[t] = w' v_[t-l] + \epsilon_[t]}

\eqn{v_[t] = F v_[t-l] + g \epsilon_[t]}

Where \eqn{v_[t]} is a state vector (defined using \code{order}) and \eqn{l} is a vector of \code{lags}.
}
\value{
  The list of the following values is returned:

  \itemize{
    \item \code{persistence} - the persistence vector.
    \item \code{transition} - matrix F.
    \item \code{measurement} - matrix w.
    \item \code{states} - the matrix of the fuzzy components of GES, where \code{rows} correspond to time and \code{cols} to states.
    \item \code{initial} - the initial values of the state vector.
    \item \code{fitted} - the fitted values of ETS.
    \item \code{forecast} - the point forecast of ETS.
    \item \code{lower} - the lower bound of prediction interval. When \code{intervals=FALSE} then NA is returned.
    \item \code{upper} - the higher bound of prediction interval. When \code{intervals=FALSE} then NA is returned.
    \item \code{residuals} - the residuals of the estimated model.
    \item \code{errors} - The matrix of 1 to h steps ahead errors.
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
\code{\link[forecast]{ets}, \link[smooth]{es}, \link[smooth]{ces}, \link[smooth]{sim.ets}}
}

\examples{
ges(rnorm(118,100,3),order=c(1),lags=c(1),h=18,holdout=TRUE,bounds=TRUE)->test

ges(rnorm(118,100,3),order=c(2,1),lags=c(1,4),h=18,holdout=TRUE,bounds=TRUE)->test

ges(rnorm(118,100,3),order=c(1,1,1),lags=c(1,3,5),h=18,holdout=TRUE,bounds=TRUE,intervals=T)->test

ges(rnorm(118,100,3),order=c(1),lags=c(1),h=18,holdout=TRUE,bounds=FALSE,CF.type="TLV")->test

ges(rnorm(118,100,3),order=c(1),lags=c(1),h=18,holdout=TRUE,bounds=FALSE,CF.type="GV")->test
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{ ges }
\keyword{ Exponential Smoothing }% __ONLY ONE__ keyword per line
