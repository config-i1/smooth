\name{ssarima}
\alias{ssarima}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
State-Space ARIMA
}
\description{
Function constructs State-Space ARIMA, estimating AR, MA terms and initial states.
}
\usage{
ssarima(data, ar.orders=c(0), i.orders=c(1), ma.orders=c(1), lags=c(1),
 constant=FALSE, initial=c("backcasting","optimal"), persistence=NULL, transition=NULL,
 CF.type=c("MSE","MAE","HAM","MLSTFE","TFL","MSTFE","MSEh"),
 h=10, holdout=FALSE, intervals=FALSE, int.w=0.95,
 int.type=c("parametric","semiparametric","nonparametric","asymmetric"),
 intermittent=c("none","simple","croston","tsb"),
 bounds=c("admissible","none"), FI=FALSE, silent=FALSE, legend=TRUE,
 xreg=NULL, go.wild=FALSE, persistenceX=NULL, transitionX=NULL, ...)
}
%- maybe also 'usage' for other objects documented here.
\arguments{

  \item{data}{ the data that needs to be forecasted.
  }
  \item{ar.orders}{ The vector of autoregressive orders, corresponding to seasonal lags. For example \code{ar.orders=c(1,1)} with \code{lags=c(1,12)} will lead to model with AR(1), SAR(1).
  }
  \item{i.orders}{ The vector of differences, corresponding to seasonal lags. For example \code{i.orders=c(0,1)} with \code{lags=c(1,12)} will lead to model with I(0), SI(1).
  }
  \item{ma.orders}{ The vector of moving average orders, corresponding to seasonal lags. For example \code{i.orders=c(1,2)} with \code{lags=c(1,12)} will lead to model with MA(1), SMA(2).
  }
  \item{lags}{ Defines lags for the corresponding orders (see examples above). The length of \code{lags} must correspond to the length of either \code{ar.orders} or \code{i.orders} or \code{ma.orders}. There is no restrictions on the length of \code{lags} vector.
  }
  \item{constant}{ If \code{TRUE}, the constant term is included in the model.
  }
  \item{initial}{ Can be either character or a vector of initial states. If it is character, then it can be \code{"optimal"}, meaning that the initial states are optimised, or \code{"backcasting"}, meaning that the initials are produced using backcasting procedure.
  }
  \item{persistence}{ Persistence vector \eqn{g}, containing smoothing parameters. Can be used if the model has been already estimated on te data before. If \code{NULL}, then estimated.
  }
  \item{transition}{ Transition matrix \eqn{F}. Can be provided as a vector. Matrix will be formed using the default \code{matrix(transition,nc,nc)}, where \code{nc} is the number of components in state vector. Can be used if the model has been already estimated on te data before. If \code{NULL}, then estimated.
  }
  \item{CF.type}{ Type of Cost Function used in optimization. \code{CF.type} can be: \code{MSE} (Mean Squared Error), \code{MAE} (Mean Absolute Error), \code{HAM} (Half Absolute Moment), \code{TFL} - Trace forecast likelihood, \code{MLSTFE} - Mean Log Squared Trace Forecast Error, \code{MSTFE} - Mean Squared Trace Forecast Error and \code{MSEh} - optimisation using only h-steps ahead error. If \code{CF.type!="MSE"} or \code{CF.type!="TFL"}, then the likelihood and model selection is done based the equivalent \code{MSE} or \code{TFL} respectively.
  }
  \item{h}{ the forecasting horizon.
  }
  \item{holdout}{ if \code{TRUE}, the holdout of the size \code{h} is taken from the end of the data.
  }
  \item{intervals}{ If \code{TRUE}, the prediction interval is constructed.
  }
  \item{int.w}{ Defines the width of the prediction interval.
  }
  \item{int.type}{ The type of intervals to construct. The first letter can be used instead of the whole word. \code{parametric} use the state-space structure of ETS, \code{semiparametric} are based on the covariance matrix of 1 to h steps ahead errors and normal distribution, \code{nonparametric} use the quantiles of the errors in that matrix, \code{asymmetric} are based on half moment of distribution.
  }
  \item{intermittent}{ Defines type of intermittent model used. Can be: 1. \code{none}, meaning that the data should be considered as a slow-moving; 2. \code{simple}, taking into account Bernoulli distribution of demand occurancies; 3. \code{croston}, based on Croston, 1972 method with SBA correction; 4. \code{tsb}, based on Teunter et al., 2011 method. The first letter can be used instead of the full words.
  }
  \item{bounds}{ What type of bounds to use for the smoothing parameters. The first letter can be used instead of the whole word.
  }
  \item{FI}{ If \code{TRUE}, Fisher Information is calculated and produced in the output.
  }
  \item{silent}{ if \code{TRUE}, no output or graph is given.
  }
  \item{legend}{ if \code{TRUE}, the legend is drawn on the output plot.
  }
  \item{xreg}{ The vector / matrix / dataframe of exogenous variables.
  }
  \item{go.wild}{ if \code{TRUE}, the transition matrix for exogenous variables is estimated, introducing non-linear interractions between the parameters. Prerequisite - non-NULL \code{xreg}.
  }
  \item{persistenceX}{ Persistence vector \eqn{g_X}, containing smoothing parameters for exogenous variables. If \code{NULL}, then estimated. Prerequisite - non-NULL \code{xreg}.
  }
  \item{transitionX}{ Transition matrix \eqn{F_x} for exogenous variables. Can be provided as a vector. Matrix will be formed using the default \code{matrix(transition,nc,nc)}, where \code{nc} is the number of components in state vector. If \code{NULL}, then estimated. Prerequisite - non-NULL \code{xreg}.
  }
  \item{...}{ All sorts of additional parameters. NOT IMPLEMENTED YET!
  }
}
\details{
The function constructs ARIMA in the Single Source of Error State-space form:

\eqn{y_[t] = o_[t] (w' v_[t-l] + x_t a_[t-1] + \epsilon_[t])}

\eqn{v_[t] = F v_[t-1] + g \epsilon_[t]}

\eqn{a_[t] = F_[X] a_[t-1] + g_[X] \epsilon_[t] / x_[t]}

Where \eqn{o_[t]} is Bernoulli distributed random variable (in case of normal data equal to 1), \eqn{v_[t]} is a state vector (defined using \code{ar.orders} and \code{i.orders}), \eqn{x_t} vector of exogenous parameters.

Due to the flexibility of the model, multiple seasonalities can be used. For example, something crazy like this can be constructed: SARIMA(1,1,1)(0,1,1)[24](2,0,1)[24*7](0,0,1)[24*30], but the estimation may take a lot of time...
}
\value{
  The list of the following values is returned:

  \itemize{
    \item \code{model} - the name of the estimated model.
    \item \code{states} - the matrix of the fuzzy components of ssarima, where \code{rows} correspond to time and \code{cols} to states.
    \item \code{initial} - the initial values of the state vector (extracted from \code{states}).
    \item \code{transition} - matrix F.
    \item \code{persistence} - the persistence vector.
    \item \code{AR} - the matrix of coefficients of AR terms.
    \item \code{I} - the matrix of coefficients of I terms.
    \item \code{MA} - the matrix of coefficients of MA terms.
    \item \code{constant} - the value of the constant term.
    \item \code{fitted} - the fitted values of ETS.
    \item \code{forecast} - the point forecast of ETS.
    \item \code{lower} - the lower bound of prediction interval. When \code{intervals=FALSE} then NA is returned.
    \item \code{upper} - the higher bound of prediction interval. When \code{intervals=FALSE} then NA is returned.
    \item \code{residuals} - the residuals of the estimated model.
    \item \code{errors} - The matrix of 1 to h steps ahead errors.
    \item \code{actuals} - the original data.
    \item \code{holdout} - the holdout part of the original data.
    \item \code{xreg} - the provided vector or matrix of exogenous variables.
    \item \code{persistenceX} - persistence vector g for exogenous variables.
    \item \code{transitionX} - transition matrix F for exogenous variables.
    \item \code{ICs} - values of information criteria of the model. Includes AIC, AICc and BIC.
    \item \code{CF} - Cost function value.
    \item \code{CF.type} - Type of cost function used in the estimation.
    \item \code{FI} - Fisher Information estimated in the optimal point.
    \item \code{accuracy} - the vector or accuracy measures. Includes MPE, MAPE, SMAPE, MASE and MASALE. Available only when \code{holdout=TRUE}.
  }
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
\code{\link[forecast]{ets}, \link[smooth]{es}, \link[smooth]{ces}, \link[smooth]{sim.ets}, \link[smooth]{ges}}
}

\examples{
# ARIMA(1,0,0)
ssarima(rnorm(118,100,3),ar.orders=c(1),lags=c(1),h=18,holdout=TRUE,intervals=TRUE)->test

# SARIMA(1,0,0)(1,0,0)[4]
ssarima(rnorm(118,100,3),ar.orders=c(2,1),lags=c(1,4),h=18,holdout=TRUE)->test

# SARIMA(1,1,1)(0,0,1)[4]
ssarima(rnorm(118,100,3),ar.orders=c(1),i.orders=c(1),ma.orders=c(1,1),
        lags=c(1,4),h=18,holdout=TRUE,bounds="n")->test
ssarima(rnorm(118,100,3),ar.orders=c(1),i.orders=c(1),ma.orders=c(1,1),
        lags=c(1,4),h=18,holdout=TRUE,bounds="n",initial="o")->test

# ARIMA(1,0,0) with trace forecast estimation
ssarima(rnorm(118,100,3),ar.orders=1,lags=1,h=18,holdout=TRUE,bounds="n",CF.type="trace")->test
ssarima(rnorm(118,100,3),ar.orders=1,lags=1,h=18,holdout=TRUE,bounds="n",CF.type="GV")->test
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{ SARIMA }
\keyword{ ARIMA }% __ONLY ONE__ keyword per line
