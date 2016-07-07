\name{auto.ces}
\alias{auto.ces}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
  Complex Exponential Smoothing Auto
}
\description{
  Function estimates CES in state-space form with information potential equal to errors with different seasonality types and chooses the one with the lowest IC value.
}
\usage{
auto.ces(data, C=c(1.1, 1), models=c("N","S","P","F"),
         initial=c("backcasting","optimal"), IC=c("CIC","AIC","AICc","BIC"),
         CF.type=c("MSE","MAE","HAM","trace","GV","TV","MSEh"),
         h=10, holdout=FALSE, intervals=FALSE, int.w=0.95,
         int.type=c("parametric","semiparametric","nonparametric","asymmetric"),
         intermittent=FALSE,
         bounds=c("none","admissible"),
         silent=c("none","all","graph","legend","output"),
         xreg=NULL, go.wild=FALSE, ...)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{data}{
    Either numeric vector or time series vector.
  }
  \item{C}{
    The initial value of complex smoothing parameter, where C[1] = a0, C[2] = a1.
  }
  \item{models}{
    The vector containing several types of seasonality that should be used in CES selection. See \link[smooth]{ces} for more details about the possible types of seasonal models.
  }
  \item{initial}{
    Can be either character or a vector of initial states. If it is character, then it can be \code{"optimal"}, meaning that the initial states are optimised, or \code{"backcasting"}, meaning that the initials are produced using backcasting procedure.
  }
  \item{IC}{
    The information criterion used in the model selection procedure. CIC - "Complex IC" is prefered due to the features of complex components etimation.
  }
  \item{CF.type}{
  Type of Cost Function used in optimization. \code{CF.type} can be: \code{MSE} (Mean Squared Error), \code{MAE} (Mean Absolute Error), \code{HAM} (Half Absolute Moment), \code{GV} - Generalized Variance, \code{trace} - Total Logarithmic Variation, \code{TV} - Total Variation and \code{MSEh} - optimisation using only h-steps ahead error. If \code{CF.type!="MSE"} or \code{CF.type!="GV"}, then the likelihood and model selection is done based the equivalent \code{MSE} or \code{GV} respectively.
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
  \item{int.type}{
    The type of intervals to construct. The first letter can be used instead of the whole word.

    \itemize{
        \item \code{parametric} use the state-space structure of ETS. For multiplicative models they are approximated using the same function as for additive. As a result they are a bit wider than should be but are still efficient. In case of mixed models this is done using simulations, which may take longer time than for the pure additive and pure multiplicative models.

        \item \code{semiparametric} are based on the covariance matrix of 1 to h steps ahead errors and normal distribution.

        \item \code{nonparametric} intervals use the values from a quantile regression on the error matrix (see Taylor and Bunn, 1999). The model used in this process is e[j] = a j^b, where j={1,..,h}.

        \item Finally \code{asymmetric} are based on half moment of distribution.
    }
  }
  \item{intermittent}{
    If \code{TRUE}, then the data is considered as intermittent, taking into account Bernoulli distribution of demand occurancies.
  }
  \item{bounds}{
    What type of bounds to use for the smoothing parameters. The first letter can be used instead of the whole word.
  }
  \item{silent}{
    If \code{silen="none"}, then nothing is silent, everything is printed out and drawn. \code{silen="all"} means that nothing is produced or drawn (except for warnings). In case of \code{silent="graph"}, no graph is produced. If \code{silen="legend"}, then the legend of the graph is skipped. And finally \code{silent="outpu"} means that nothing is printed out in the console, but the graph is produced. the variable also accepts \code{TRUE} and \code{FALSE}. In this case \code{silent=TRUE} is equivalent to \code{silent="all"}, while \code{silent=FALSE} is equivalent to \code{silent="none"}. The variable also accepts first letter of words ("n", "a", "g", "l", "o").
  }
  \item{xreg}{
    The vector (either numeric or time series) or matrix (or data.frame) of exogenous variables that should be included in the model. If matrix included than columns should contain variables and rows - observations. Note that \code{xreg} should have the number of observations equal either to in-sample or to the whole series. If the number of observations in \code{xreg} is equal to in-sample, then the values in the holdout sample are forecasted using Naive.
  }
  \item{go.wild}{
    If \code{TRUE}, the transition matrix for exogenous variables is estimated, introducing non-linear interractions between the parameters. Prerequisite - non-NULL \code{xreg}.
  }
  \item{...}{
    Other non-documented parameters. For example \code{FI=TRUE} will make the function also produce Fisher Information matrix, which then can be used to calculated variances of parameters of the model.
  }
}
\details{
  The function estimates several Complex Exponential Smoothing in the state-space 2 described in Svetunkov, Kourentzes (2015) with the information potential equal to the approximation error using different types of seasonality and chooses the one with the lowest value of information criterion.
}
\value{
  The list of the returned values is the same as in \link[smooth]{ces} function.
}
\references{
    \itemize{
        \item{Svetunkov, I., Kourentzes, N. (February 2015). Complex exponential smoothing. Working Paper of Department of Management Science, Lancaster University 2015:1, 1-31.}
        \item{Svetunkov I., Kourentzes N. (2015) Complex Exponential Smoothing for Time Series Forecasting. Not yet published.}
        \item{Hyndman, R.J., Koehler, A.B., Ord, J.K., and Snyder, R.D. (2008) Forecasting with exponential smoothing: the state space approach, Springer-Verlag. \url{http://www.exponentialsmoothing.net}.}
        \item{Svetunkov S. (2012) Complex-Valued Modeling in Economics and Finance. SpringerLink: Bucher. Springer.}
    }
}
\author{
  Ivan Svetunkov, \email{ivan@svetunkov.ru}
}
\seealso{
    \code{\link[smooth]{ces}, \link[forecast]{ets}, \link[forecast]{forecast}, \link[stats]{ts}}
}
\examples{
y <- rnorm(100,10,3)
y <- ts(y,frequency=12)
y_ces_holdout <- auto.ces(y,h=20,holdout=TRUE)
y_ces_no_holdout <- auto.ces(y,h=20,holdout=FALSE)

library("Mcomp")
y <- ts(c(M3$N0740$x,M3$N0740$xx),start=start(M3$N0740$x),frequency=frequency(M3$N0740$x))
N0740_ces <- auto.ces(y,h=8,holdout=TRUE,models=c("N","P"),intervals=TRUE,int.w=0.8,IC="AIC")

y <- ts(c(M3$N1683$x,M3$N1683$xx),start=start(M3$N1683$x),frequency=frequency(M3$N1683$x))
N1683_ces_S <- auto.ces(y,h=18,holdout=TRUE,intervals=TRUE)
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{ ces }
\keyword{ complex exponential smoothing }
\keyword{ exponential smoothing }
\keyword{ forecasting }
\keyword{ complex variables }
