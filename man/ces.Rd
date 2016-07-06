\name{ces}
\alias{ces}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
  Complex Exponential Smoothing
}
\description{
  Function estimates CES in state-space form with information potential equal to errors and returns several variables.
}
\usage{
ces(data, C=c(1.1, 1), seasonality=c("N","S","P","F"),
    initial=c("backcasting","optimal"),
    CF.type=c("MSE","MAE","HAM","MLSTFE","TFL","MSTFE","MSEh"),
    h=10, holdout=FALSE, intervals=FALSE, int.w=0.95,
    int.type=c("parametric","semiparametric","nonparametric","asymmetric"),
    intermittent=FALSE,
    bounds=c("none","admissible"),
    silent=c("none","all","graph","legend","output"),
    xreg=NULL, initialX=NULL, go.wild=FALSE, persistenceX=NULL, transitionX=NULL, ...)
}
\arguments{
  \item{data}{
    Either numeric vector or time series vector.
  }
  \item{C}{
    The initial value of complex smoothing parameter, where C[1] = a0, C[2] = a1.
  }
  \item{seasonality}{
    The type of seasonality used in CES. Can be: "N" - No seasonality; "S" - Simple seasonality, using lagged CES (based on \code{t-m} observation, where \code{m} is the seasonality lag); "P" - Partial seasonality with real seasonal components (equivalent to additive seasonality); "F" - Full seasonality with complex seasonal components (can do both multiplicative and additive seasonality, depending on the data).
    Any seasonal CES can only be constructed for time series vectors.
  }
  \item{initial}{
    Can be either character or a vector of initial states. If it is character, then it can be \code{"optimal"}, meaning that the initial states are optimised, or \code{"backcasting"}, meaning that the initials are produced using backcasting procedure.
  }
  \item{CF.type}{ Type of Cost Function used in optimization. \code{CF.type} can be: \code{MSE} (Mean Squared Error), \code{MAE} (Mean Absolute Error), \code{HAM} (Half Absolute Moment), \code{TFL} - Trace forecast likelihood, \code{MLSTFE} - Mean Log Squared Trace Forecast Error, \code{MSTFE} - Mean Squared Trace Forecast Error and \code{MSEh} - optimisation using only h-steps ahead error. If \code{CF.type!="MSE"} or \code{CF.type!="TFL"}, then the likelihood and model selection is done based the equivalent \code{MSE} or \code{TFL} respectively.
  }
  \item{h}{
    The forecasting horizon.
  }
  \item{holdout}{
    If \code{TRUE}, the holdout sample of size h will be taken from the data. If \code{FALSE}, no holdout is defined.
  }
  \item{intervals}{
    If \code{TRUE}, the prediction intervals are constructed.
  }
  \item{int.w}{
    The width of prediction intervals. Default is 95\%.
  }
  \item{int.type}{ The type of intervals to construct. The first letter can be used instead of the whole word. \code{parametric} uses the state-space structure of ETS, \code{semiparametric} is based on the covariance matrix of 1 to h steps ahead errors and normal distribution, \code{nonparametric} uses the values from a quantile regression on the error matrix (see Taylor and Bunn, 1999), \code{asymmetric} are based on half moment of distribution. PARTLY IMPLEMENTED - PARAMETRIC INTERVALS FOR MULTIPLICATIVE TREND / SEASONALITY ARE NOT THERE YET!
  }
  \item{intermittent}{ If \code{TRUE}, then the data is considered as intermittent, taking into account Bernoulli distribution of demand occurancies.
  }
  \item{bounds}{ What type of bounds to use for the smoothing parameters. The first letter can be used instead of the whole word.
  }
  \item{silent}{
    If \code{silen="none"}, then nothing is silent, everything is printed out and drawn. \code{silen="all"} means that nothing is produced or drawn (except for warnings). In case of \code{silent="graph"}, no graph is produced. If \code{silen="legend"}, then the legend of the graph is skipped. And finally \code{silent="outpu"} means that nothing is printed out in the console, but the graph is produced. the variable also accepts \code{TRUE} and \code{FALSE}. In this case \code{silent=TRUE} is equivalent to \code{silent="all"}, while \code{silent=FALSE} is equivalent to \code{silent="none"}. The variable also accepts first letter of words ("n", "a", "g", "l", "o").
  }
  \item{xreg}{
    The vector (either numeric or time series) or matrix (or data.frame) of exogenous variables that should be included in the model. If matrix included than columns should contain variables and rows - observations. Note that \code{xreg} should have the number of observations equal either to in-sample or to the whole series. If the number of observations in \code{xreg} is equal to in-sample, then the values in the holdout sample are forecasted using Naive.
  }
  \item{initialX}{
    The vector of initial parameters for exogenous variables. Ignored if \code{xreg} is NULL.
  }
  \item{go.wild}{
    If \code{TRUE}, the transition matrix for exogenous variables is estimated, introducing non-linear interractions between the parameters. Prerequisite - non-NULL \code{xreg}.
  }
  \item{persistenceX}{
    Persistence vector \eqn{g_X}, containing smoothing parameters for exogenous variables. If \code{NULL}, then estimated. Prerequisite - non-NULL \code{xreg}.
  }
  \item{transitionX}{
    Transition matrix \eqn{F_x} for exogenous variables. Can be provided as a vector. Matrix will be formed using the default \code{matrix(transition,nc,nc)}, where \code{nc} is the number of components in state vector. If \code{NULL}, then estimated. Prerequisite - non-NULL \code{xreg}.
  }
  \item{...}{
    Other non-documented parameters. For example \code{FI=TRUE} will make the function also produce Fisher Information matrix, which then can be used to calculated variances of parameters of the model.
  }
}
\details{
  The function estimates Complex Exponential Smoothing in the state-space 2 described in Svetunkov, Kourentzes (2015) with the information potential equal to the approximation error.
  The estimation of initial states of xt is done using backcast.
}
\value{
  The list of the following values is returned:

  \itemize{
    \item \code{model} - type of constructed model.
    \item \code{states} - the matrix of the components of CES. The included minimum is "level" and "potential". In the case of seasonal model the seasonal component is also included. In the case of exogenous variables the estimated coefficients for the exogenous variables are also included.
    \item \code{A} - complex smoothing parameter in the form a0 + ia1
    \item \code{B} - smoothing parameter for the seasonal component. Can either be real (if \code{seasonality="P"}) or complex (if \code{seasonality="F"}) in a form b0 + ib1.
    \item \code{fitted} - the fitted values of CES.
    \item \code{forecast} - the point forecast of CES.
    \item \code{lower} - the lower bound of prediction interval. When \code{intervals=FALSE} then NA is returned.
    \item \code{upper} - the upper bound of prediction interval. When \code{intervals=FALSE} then NA is returned.
    \item \code{residuals} - the residuals of the estimated model.
    \item \code{errors} - The matrix of 1 to h steps ahead errors.
    \item \code{actuals} - The data provided in the call of the function.
    \item \code{holdout} - the holdout part of the original data.
    \item \code{xreg} - the provided vector or matrix of exogenous variables.
    \item \code{initialX} - initial values for parameters of exogenous variables.
    \item \code{persistenceX} - persistence vector g for exogenous variables.
    \item \code{transitionX} - transition matrix F for exogenous variables.
    \item \code{ICs} - values of information criteria of the model. Includes AIC, AICc, BIC and CIC (Complex IC).
    \item \code{CF} - Cost function value.
    \item \code{CF.type} - Type of cost function used in the estimation.
    \item \code{FI} - Fisher Information. Equal to NULL if \code{FI=FALSE} or when \code{FI} is not provided at all.
    \item \code{accuracy} - the vector or accuracy measures for the holdout sample. Includes MPE, MAPE, SMAPE, MASE, MAE/mean, RelMAE and Bias coefficient (based on complex numbers). Available only when \code{holdout=TRUE}.
  }
}
\references{
    \itemize{
        \item{Svetunkov, I., Kourentzes, N. (February 2015). Complex exponential smoothing. Working Paper of Department of Management Science, Lancaster University 2015:1, 1-31.}
        \item{Hyndman, R.J., Koehler, A.B., Ord, J.K., and Snyder, R.D. (2008) Forecasting with exponential smoothing: the state space approach, Springer-Verlag. \url{http://www.exponentialsmoothing.net}.}
        \item{Svetunkov S. (2012) Complex-Valued Modeling in Economics and Finance. SpringerLink: Bucher. Springer.}
    }
}
\author{
 Ivan Svetunkov, \email{ivan@svetunkov.ru}
}
\seealso{
\code{\link[forecast]{ets}, \link[forecast]{forecast}, \link[stats]{ts}, \link[smooth]{auto.ces}}
}
\examples{
y <- rnorm(100,10,3)
y_ces_holdout <- ces(y,h=20,holdout=TRUE)
y_ces_no_holdout <- ces(y,h=20,holdout=FALSE)

y <- 500 - c(1:100)*0.5 + rnorm(100,10,3)
y_ces <- ces(y,h=20,holdout=TRUE,intervals=TRUE,bounds="a")

library("Mcomp")
y <- ts(c(M3$N0740$x,M3$N0740$xx),start=start(M3$N0740$x),frequency=frequency(M3$N0740$x))
N0740_ces <- ces(y,h=8,holdout=TRUE,seasonality="S",intervals=TRUE,int.w=0.8)

y <- ts(c(M3$N1683$x,M3$N1683$xx),start=start(M3$N1683$x),frequency=frequency(M3$N1683$x))
N1683_ces_S <- ces(y,h=18,holdout=TRUE,seasonality="S",intervals=TRUE)
N1683_ces_P <- ces(y,h=18,holdout=TRUE,seasonality="P",intervals=TRUE)
N1683_ces_F <- ces(y,h=18,holdout=TRUE,seasonality="F",intervals=TRUE)

x <- cbind(c(rep(0,25),1,rep(0,43)),c(rep(0,10),1,rep(0,58)))
ces(ts(c(M3$N1457$x,M3$N1457$xx),frequency=12),h=18,holdout=TRUE,
    intervals=TRUE,xreg=x,CF.type="MSTFE")->test
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{ ces }
\keyword{ complex exponential smoothing }
\keyword{ exponential smoothing }
\keyword{ forecasting }
\keyword{ complex variables }
