\name{es}
\alias{es}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
Exponential Smoothing in SSOE state-space model
}
\description{
Function constructs ETS model and returns forecast, fitted values, errors and matrix of states.
}
\usage{
es(data, model="ZZZ", persistence=NULL, phi=NULL,
   initial=NULL, initial.season=NULL, IC=c("AICc","AIC","BIC"),
   persistenceX=NULL, transitionX=NULL,
   CF.type=c("MSE","MAE","HAM","trace","GV","TV","MSEh"),
   FI=FALSE, intervals=FALSE, int.w=0.95,
   int.type=c("parametric","semiparametric","nonparametric","asymmetric"),
   bounds=c("usual","admissible","none"), holdout=FALSE, h=10, silent=FALSE, legend=TRUE,
   xreg=NULL, go.wild=FALSE, intermittent=FALSE, ...)

}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{data}{ The vector or ts object, containing the data needed to be forecasted.
  }
  \item{model}{ The type of ETS model. Can consist of 3 or 4 chars: \code{ANN}, \code{AAN}, \code{AAdN}, \code{AAA}, \code{AAdA}, \code{MAdM} etc. \code{ZZZ} means that the model will be selected based on the chosen IC type. If \code{model="CCC"}, then all the models are estimated and the combination of their forecasts using AIC weights if produced (Kolassa, 2011). This can also be regulated. For example, \code{model="CCN"} will combine forecasts of all the non-seasonal models.

  The parameter \code{model} can also be a vector of names of models for a finer tuning (pool of models). For example, \code{model=c("ANN","AAA")} will estimate only two models and select the best of them.

  Keep in mind that when model selection or combination is selected, the optimiser may return not as accurate estimates of parameters as when individual model is selected. This may happen because of the decreased precision of optimiser, which is done intentionally to speed up the optimisation process. Furthermore, model selection uses Branch and Bound method and may skip some models that could have slightly smaller information criteria.
  }
  \item{persistence}{ The vector of smoothing parameters. If \code{NULL}, the parameters are estimated.
  }
  \item{phi}{ The value of damping parameter. If \code{NULL} then the parameter is estimated.
  }
  \item{initial}{ The vector of initial values for level and trend. If \code{NULL}, it is estimated in the optimization.
  }
  \item{initial.season}{ The vector of initial values for seasonal components. If \code{NULL}, they are estimated in the optimization.
  }
  \item{IC}{ The information criterion to use in the model selection.
  }
  \item{persistenceX}{ Persistence vector \eqn{g_X}, containing smoothing parameters for exogenous variables. If \code{NULL}, then estimated. Prerequisite - non-NULL \code{xreg}.
  }
  \item{transitionX}{ Transition matrix \eqn{F_x} for exogenous variables. Can be provided as a vector. Matrix will be formed using the default \code{matrix(transition,nc,nc)}, where \code{nc} is the number of components in state vector. If \code{NULL}, then estimated. Prerequisite - non-NULL \code{xreg}.
  }
  \item{CF.type}{ Type of Cost Function used in optimization. \code{CF.type} can be: \code{MSE} (Mean Squared Error), \code{MAE} (Mean Absolute Error), \code{HAM} (Half Absolute Moment), \code{GV} - Generalized Variance, \code{trace} - Total Logarithmic Variation, \code{TV} - Total Variation and \code{MSEh} - optimisation using only h-steps ahead error. If \code{CF.type!="MSE"} or \code{CF.type!="GV"}, then the likelihood and model selection is done based the equivalent \code{MSE} or \code{GV} respectively.
  }
  \item{FI}{ If \code{TRUE}, Fisher Information is calculated and produced in the output.
  }
  \item{intervals}{ If \code{TRUE}, the prediction interval is constructed.
  }
  \item{int.w}{ Defines the width of the prediction interval.
  }
  \item{int.type}{ The type of intervals to construct. The first letter can be used instead of the whole word.

  \itemize{
    \item \code{parametric} use the state-space structure of ETS. For multiplicative models they are approximated using the same function as for additive. As a result they are a bit wider than should be but are still efficient. In case of mixed models this is done using simulations, which may take longer time than for the pure additive and pure multiplicative models.

    \item \code{semiparametric} are based on the covariance matrix of 1 to h steps ahead errors and normal distribution.

    \item \code{nonparametric} intervals use the values from a quantile regression on the error matrix (see Taylor and Bunn, 1999).

    \item Finally \code{asymmetric} are based on half moment of distribution.
  }
  }
  \item{bounds}{ What type of bounds to use for the smoothing parameters. The first letter can be used instead of the whole word.
  }
  \item{holdout}{ If \code{TRUE}, the holdout sample of size \code{h} is taken from the end of the data.
  }
  \item{h}{ The length of the forecasting horizon.
  }
  \item{silent}{ If \code{TRUE}, no output or graphs are printed out.
  }
  \item{legend}{ If \code{TRUE}, the legend is drawn in the outplot.
  }
  \item{xreg}{ The vector / matrix / dataframe of exogenous variables.
  }
  \item{go.wild}{ if \code{TRUE}, the transition matrix for exogenous variables is estimated, introducing non-linear interractions between the parameters. Prerequisite - non-NULL \code{xreg}.
  }
  \item{intermittent}{ If \code{TRUE}, then the data is considered as intermittent, taking into account Bernoulli distribution of demand occurancies.
  }
  \item{...}{ Other parameters that are passed to DON'T KNOW WHAT YET.
  }
}
\details{
The function estimates ETS in a form of the Single Source of Error State-space model of the following type:

\eqn{y_[t] = o_[t] (w(v_[t-l]) + x_t a_[t-1] + r(v_[t-l]) \epsilon_[t])}

\eqn{v_[t] = f(v_[t-l]) + g(v_[t-l]) \epsilon_[t]}

\eqn{a_[t] = F_[X] a_[t-1] + g_[X] \epsilon_[t] / x_[t]}

Where \eqn{o_[t]} is Bernoulli distributed random variable (in case of normal data equal to 1), \eqn{v_[t]} is a state vector (defined using \code{orders}) and \eqn{l} is a vector of \code{lags}, \eqn{x_t} vector of exogenous parameters. w(.) is the measurement function, r(.) is an error function, f(.) is transition function and g(.) is persistence function.
}
\value{
  The list of the following values is returned for the simpler models:

  \itemize{
    \item \code{model} - type of constructed model.
    \item \code{persistence} - the persistence vector.
    \item \code{phi} - the value of damping parameter.
    \item \code{states} - the matrix of the components of CES. The included minimum is "level" and "potential". In the case of seasonal model the seasonal component is also included. In the case of exogenous variables the estimated coefficients for the exogenous variables are also included.
    \item \code{initial} - the intial values of the state vector (non-seasonal).
    \item \code{initial.season} - the intial values of the seasonal part of state vector.
    \item \code{fitted} - the fitted values of ETS.
    \item \code{forecast} - the point forecast of ETS.
    \item \code{lower} - the lower bound of prediction interval. When \code{intervals=FALSE} then NA is returned.
    \item \code{upper} - the higher bound of prediction interval. When \code{intervals=FALSE} then NA is returned.
    \item \code{residuals} - the residuals of the estimated model.
    \item \code{errors} - trace forecast in-sample errors, returned as a matrix. In the case of trace forecasts this is the matrix used in optimisation. In non-trace estimations it is returned just for the information.
    \item \code{actuals} - the original data.
    \item \code{holdout} - the holdout part of the original data.
    \item \code{xreg} - the provided vector or matrix of exogenous variables.
    \item \code{persistenceX} - persistence vector g for exogenous variables.
    \item \code{transitionX} - transition matrix F for exogenous variables.
    \item \code{ICs} - values of information criteria of the model. Includes AIC, AICc and BIC.
    \item \code{CF} - Cost function value.
    \item \code{CF.type} - Type of cost function used in the estimation.
    \item \code{FI} - Fisher Information estimated in the optimal point.
    \item \code{xreg} - the provided vector or matrix of exogenous variables.
    \item \code{accuracy} - the vector or accuracy measures for the holdout sample. Includes MPE, MAPE, SMAPE, MASE and MASALE. Available only when \code{holdout=T}.
  }

  If the combination of forecasts is produced (using \code{model="CCC"}), then the shorter list of the values is returned:

  \itemize{
    \item \code{model},
    \item \code{fitted},
    \item \code{forecast},
    \item \code{lower},
    \item \code{upper},
    \item \code{residuals},
    \item \code{actuals},
    \item \code{holdout},
    \item \code{ICs} - the IC weights used in the combination.
    \item \code{CF.type},
    \item \code{xreg},
    \item \code{accuracy}.
  }
}
\references{
    \enumerate{
        \item{Hyndman, R.J., Koehler, A.B., Ord, J.K., and Snyder, R.D. (2008) Forecasting with exponential smoothing: the state space approach, Springer-Verlag. \url{http://www.exponentialsmoothing.net}.}
        \item{Taylor, J.W. and Bunn, D.W. (1999) A Quantile Regression Approach to Generating Prediction Intervals. Management Science, Vol 45, No 2, pp 225-237.}
        \item{Kolassa, S. (2011) Combining exponential smoothing forecasts using Akaike weights. International Journal of Forecasting, 27, pp 238 - 251.}
    }
}
\author{
Ivan Svetunkov, \email{ivan@svetunkov.ru}
}

\seealso{
\code{\link[forecast]{ets}, \link[forecast]{forecast}, \link[stats]{ts}, \link[smooth]{sim.ets}}
}
\examples{
library(Mcomp)

# See how holdout and trace parameters influence the forecast
es(M3$N1245$x,model="AAdN",h=8,holdout=FALSE,trace=FALSE) -> test
es(M3$N1245$x,model="AAdN",h=8,holdout=FALSE,trace=TRUE) -> test
es(M3$N2568$x,model="MAM",h=18,holdout=TRUE,trace=TRUE) -> test

# Model selection example
es(M3$N1245$x,model="ZZN",IC="AIC",h=8,holdout=FALSE,bounds="a") -> test

# Model selection. Compare AICc of these two models:
es(M3$N1683$x,"ZZZ",h=10,holdout=TRUE)->test
es(M3$N1683$x,"MAdM",h=10,holdout=TRUE)->test

# Combination example
es(M3$N1245$x,model="CCC",h=8,holdout=TRUE) -> test

# Model selection using a specified pool of models
es(M3$N1587$x,model=c("ANN","AAM","AMdA"),h=18) -> test

# Semiparametric intervals example
es(M3$N1587$x,h=18,holdout=TRUE,intervals=TRUE,int.type="s") -> test

# Exogenous variables in ETS example
x <- cbind(c(rep(0,25),1,rep(0,43)),c(rep(0,10),1,rep(0,58)))
es(ts(c(M3$N1457$x,M3$N1457$xx),frequency=12),h=18,holdout=TRUE,xreg=x,trace=TRUE)->test
}
\keyword{ exponential smoothing }
\keyword{ ETS }
\keyword{ forecasting }
\keyword{ trace likelihood }
