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
      CF.type=c("MSE","MAE","HAM","trace","GV","TV","MSEh"),
      use.test=FALSE, intervals=FALSE, int.w=0.95,
      bounds=FALSE, holdout=FALSE, h=1, silent=FALSE, legend=TRUE,
      xreg=NULL)
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
  \item{CF.type}{
  Type of Cost Function used in optimization. \code{CF.type} can be: \code{MSE} (Mean Squared Error), \code{MAE} (Mean Absolute Error), \code{HAM} (Half Absolute Moment), \code{GV} - Generalized Variance, \code{trace} - Total Logarithmic Variation, \code{TV} - Total Variation and \code{MSEh} - optimisation using only h-steps ahead error. If \code{CF.type!="MSE"} or \code{CF.type!="GV"}, then the likelihood and model selection is done based the equivalent \code{MSE} or \code{GV} respectively.
  }
  \item{use.test}{
    If \code{TRUE} the statistical test is used on the imaginary part of the complex smoothing parameter to define if it is significantly different from 1.
  }
  \item{intervals}{
    If \code{TRUE}, the prediction intervals are constructed.
  }
  \item{int.w}{
    The width of prediction intervals. Default is 95\%.
  }
  \item{bounds}{
    If \code{TRUE}, the smoothing parameter is chosen from the stability region, which makes estimation much slower. If \code{FALSE}, no bounds are used, which is fast but may result in non-stable model.
  }
  \item{holdout}{
    If \code{TRUE}, the holdout sample of size h will be taken from the data. If \code{FALSE}, no holdout is defined.
  }
  \item{h}{
    The forecasting horizon.
  }
  \item{silent}{
    If \code{TRUE}, no output is given and no plot is drawn.
  }
  \item{legend}{
    If \code{FALSE}, the legend in the graph is skiped.
  }
  \item{xreg}{
    The vector (either numeric or time series) or matrix (or data.frame) of exogenous variables that should be included in CES. If matrix included than columns should contain variables and rows - observations. xreg should have the number of observations equal either to in-sample or to the whole series. If the number of observations in xreg is equal to in-sample, xreg in the holdout sample is taken to be equal to zero. If data is ts object then xreg also must be ts.
  }
}
\details{
  The function estimates Complex Exponential Smoothing in the state-space 2 described in Svetunkov, Kourentzes (2015) with the information potential equal to the approximation error.
  The estimation of initial states of xt is done using backcast.
}
\value{
  The list of the following values is returned:

  \itemize{
    \item \code{A} - complex smoothing parameter in the form a0 + ia1
    \item \code{B} - smoothing parameter for the seasonal component. Can either be real (if \code{seasonality="P"}) or complex (if \code{seasonality="F"}) in a form b0 + ib1.
    \item \code{residuals} - the residuals of the estimated model.
    \item \code{errors} - The matrix of 1 to h steps ahead errors.
    \item \code{actuals} - The data provided in the call of the function.
    \item \code{holdout} - the holdout part of the original data.
    \item \code{fitted} - the fitted values of CES.
    \item \code{forecast} - the point forecast of CES.
    \item \code{lower} - the lower bound of prediction interval. When \code{intervals=\code{FALSE}} then NA is returned.
    \item \code{upper} - the upper bound of prediction interval. When \code{intervals=\code{FALSE}} then NA is returned.
    \item \code{states} - the matrix of the components of CES. The included minimum is "level" and "potential". In the case of seasonal model the seasonal component is also included. In the case of exogenous variables the estimated coefficients for the exogenous variables are also included.
    \item \code{ICs} - values of information criteria of the model. Includes AIC, AICc, BIC and CIC (Complex IC).
    \item \code{CF.type} - Type of cost function used in the estimation.
    \item \code{FI} - Fisher Information.
    \item \code{accuracy} - the vector or accuracy measures. Includes MAPE, MASE and MASALE. Available only when \code{holdout=T}.
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
\code{\link[forecast]{ets}, \link[forecast]{forecast}, \link[stats]{ts}}
}
\examples{
y <- rnorm(100,10,3)
y_ces_holdout <- ces(y,h=20,holdout=TRUE)
y_ces_no_holdout <- ces(y,h=20,holdout=FALSE)

y <- 500 - c(1:100)*0.5 + rnorm(100,10,3)
y_ces <- ces(y,h=20,holdout=TRUE,intervals=TRUE,bounds=TRUE)

library("Mcomp")
y <- ts(c(M3$N0740$x,M3$N0740$xx),start=start(M3$N0740$x),frequency=frequency(M3$N0740$x))
N0740_ces <- ces(y,h=8,holdout=TRUE,seasonality="S",intervals=TRUE,int.w=0.8)

y <- ts(c(M3$N1683$x,M3$N1683$xx),start=start(M3$N1683$x),frequency=frequency(M3$N1683$x))
N1683_ces_S <- ces(y,h=18,holdout=TRUE,seasonality="S",intervals=TRUE)
N1683_ces_P <- ces(y,h=18,holdout=TRUE,seasonality="P",intervals=TRUE)
N1683_ces_F <- ces(y,h=18,holdout=TRUE,seasonality="F",intervals=TRUE)

x <- cbind(c(rep(0,25),1,rep(0,43)),c(rep(0,10),1,rep(0,58)))
ces(ts(c(M3$N1457$x,M3$N1457$xx),frequency=12),h=18,holdout=TRUE,
    intervals=TRUE,xreg=x,CF.type="trace")->test
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{ ces }
\keyword{ complex exponential smoothing }
\keyword{ exponential smoothing }
\keyword{ forecasting }
\keyword{ complex variables }
