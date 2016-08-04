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
   initial=c("optimal","backcasting"), initial.season=NULL, IC=c("AICc","AIC","BIC"),
   CF.type=c("MSE","MAE","HAM","MLSTFE","TFL","MSTFE","MSEh"),
   h=10, holdout=FALSE, intervals=FALSE, int.w=0.95,
   int.type=c("parametric","semiparametric","nonparametric","asymmetric"),
   intermittent=c("none","fixed","croston","tsb","auto"),
   bounds=c("usual","admissible","none"),
   silent=c("none","all","graph","legend","output"),
   xreg=NULL, initialX=NULL, go.wild=FALSE, persistenceX=NULL, transitionX=NULL, ...)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{data}{
    The vector or ts object, containing the data needed to be forecasted.
  }
  \item{model}{
    The type of ETS model. Can consist of 3 or 4 chars: \code{ANN}, \code{AAN}, \code{AAdN}, \code{AAA}, \code{AAdA}, \code{MAdM} etc. \code{ZZZ} means that the model will be selected based on the chosen IC type. If \code{model="CCC"}, then all the models are estimated and the combination of their forecasts using AIC weights if produced (Kolassa, 2011). This can also be regulated. For example, \code{model="CCN"} will combine forecasts of all the non-seasonal models.

    The parameter \code{model} can also be a vector of names of models for a finer tuning (pool of models). For example, \code{model=c("ANN","AAA")} will estimate only two models and select the best of them.

    Keep in mind that when model selection or combination is selected, the optimiser may return not as accurate estimates of parameters as when individual model is selected. This may happen because of the decreased precision of optimiser, which is done intentionally to speed up the optimisation process. Furthermore, model selection uses Branch and Bound method and may skip some models that could have slightly smaller information criteria.
  }
  \item{persistence}{
    The vector of smoothing parameters. If \code{NULL}, the parameters are estimated.
  }
  \item{phi}{
    The value of damping parameter. If \code{NULL} then the parameter is estimated.
  }
  \item{initial}{
    Can be either character or a vector of initial states. If it is character, then it can be \code{"optimal"}, meaning that the initial states are optimised, or \code{"backcasting"}, meaning that the initials are produced using backcasting procedure. \code{initial.season} will always be estimated in the way defined by \code{initial.season}.
  }
  \item{initial.season}{
    The vector of initial values for seasonal components. If \code{NULL}, they are estimated in the optimization.
  }
  \item{IC}{
    The information criterion to use in the model selection.
  }
  \item{CF.type}{
    Type of Cost Function used in optimization. \code{CF.type} can be: \code{MSE} (Mean Squared Error), \code{MAE} (Mean Absolute Error), \code{HAM} (Half Absolute Moment), \code{TFL} - Trace forecast likelihood, \code{MLSTFE} - Mean Log Squared Trace Forecast Error, \code{MSTFE} - Mean Squared Trace Forecast Error and \code{MSEh} - optimisation using only h-steps ahead error. If \code{CF.type!="MSE"} or \code{CF.type!="TFL"}, then the likelihood and model selection is done based the equivalent \code{MSE} or \code{TFL} respectively.

    There are also available analytical approximations for multistep functions: \code{aMSEh}, \code{aMSTFE}, \code{aMLSTFE} and \code{aTFL}. These can be useful on small samples.
  }
  \item{h}{
    The length of the forecasting horizon.
  }
  \item{holdout}{
    If \code{TRUE}, the holdout sample of size \code{h} is taken from the end of the data.
  }
  \item{intervals}{
    If \code{TRUE}, the prediction interval is constructed.
  }
  \item{int.w}{
    Defines the width of the prediction interval.
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
    Defines type of intermittent model used. Can be: 1. \code{none}, meaning that the data should be considered as non-intermittent; 2. \code{fixed}, taking into account constant Bernoulli distribution of demand occurancies; 3. \code{croston}, based on Croston, 1972 method with SBA correction; 4. \code{tsb}, based on Teunter et al., 2011 method. 5. \code{auto} - automatic selection of intermittency type based on data. The first letter can be used instead of the full words.
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
    \item \code{states} - the matrix of the components of CES. The included minimum is "level" and "potential". In the case of seasonal model the seasonal component is also included. In the case of exogenous variables the estimated coefficients for the exogenous variables are also included.
    \item \code{persistence} - the persistence vector.
    \item \code{phi} - the value of damping parameter.
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
    \item \code{iprob} - the fitted and forecasted values of the probability of demand occurrence.
    \item \code{xreg} - the provided vector or matrix of exogenous variables.
    \item \code{initialX} - initial values for parameters of exogenous variables.
    \item \code{persistenceX} - persistence vector g for exogenous variables.
    \item \code{transitionX} - transition matrix F for exogenous variables.
    \item \code{ICs} - values of information criteria of the model. Includes AIC, AICc and BIC.
    \item \code{CF} - Cost function value.
    \item \code{CF.type} - Type of cost function used in the estimation.
    \item \code{FI} - Fisher Information. Equal to NULL if \code{FI=FALSE} or when \code{FI} is not provided at all.
    \item \code{accuracy} - the vector or accuracy measures for the holdout sample. Includes MPE, MAPE, SMAPE, MASE, MAE/mean, RelMAE and Bias coefficient (based on complex numbers). Available only when \code{holdout=TRUE}.
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
    \item \code{ICs} - combined IC,
    \item \code{ICw} - IC weights used in the combination,
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
        \item{Teunter R., Syntetos A., Babai Z. (2011). Intermittent demand: Linking forecasting to inventory obsolescence. European Journal of Operational Research 214, 606-615.}
        \item{Croston, J. (1972) Forecasting and stock control for intermittent demands. Operational Research Quarterly, 23(3), 289-303.}
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
es(M3$N1245$x,model="AAdN",h=8,holdout=FALSE,CF.type="MSE") -> test
es(M3$N1245$x,model="AAdN",h=8,holdout=FALSE,CF.type="TFL") -> test
es(M3$N2568$x,model="MAM",h=18,holdout=TRUE,CF.type="MSTFE") -> test

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
es(ts(c(M3$N1457$x,M3$N1457$xx),frequency=12),h=18,holdout=TRUE,xreg=x,CF.type="TFL")->test
es(ts(c(M3$N1457$x,M3$N1457$xx),frequency=12),h=18,holdout=TRUE,xreg=x,go.wild=TRUE)->test

# Intermittent data example
x <- rpois(100,0.2)
# Croston's method with the best ETS for demand sizes
es(x,"ZZN",intermittent="croston")->test
# TSB based on iETS(M,N,N)
es(x,"MNN",intermittent="tsb")->test
# Constant probability based on iETS(M,N,N)
es(x,"MNN",intermittent="fixed")->test
# Best type of intermittent model based on iETS(Z,Z,N)
es(x,"ZZN",intermittent="auto")->test
}
\keyword{ exponential smoothing }
\keyword{ ETS }
\keyword{ forecasting }
\keyword{ trace likelihood }
