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
ges(data, orders=c(1,1), lags=c(1,frequency(data)),
    persistence=NULL, transition=NULL, measurement=NULL,
    initial=c("optimal","backcasting"),
    cfType=c("MSE","MAE","HAM","MLSTFE","MSTFE","MSEh"),
    h=10, holdout=FALSE,
    intervals=c("none","parametric","semiparametric","nonparametric"), level=0.95,
    intermittent=c("none","auto","fixed","croston","tsb"),
    bounds=c("admissible","none"),
    silent=c("none","all","graph","legend","output"),
    xreg=NULL, initialX=NULL, updateX=FALSE, persistenceX=NULL, transitionX=NULL, ...)
}
%- maybe also 'usage' for other objects documented here.
\arguments{

  \item{data}{
    The data that needs to be forecasted.
  }
  \item{orders}{
    Order of the model. Specified as vector of number of states with different lags. For example, \code{orders=c(1,1)} means that there are two states: one of the first lag type, the second of the second type.
  }
  \item{lags}{
    Defines lags for the corresponding orders. If, for example, \code{orders=c(1,1)} and lags are defined as \code{lags=c(1,12)}, then the model will have two states: the first will have lag 1 and the second will have lag 12. The length of  \code{lags} must correspond to the length of \code{orders}.
  }
  \item{persistence}{
    Persistence vector \eqn{g}, containing smoothing parameters. If \code{NULL}, then estimated.
  }
  \item{transition}{
    Transition matrix \eqn{F}. Can be provided as a vector. Matrix will be formed using the default \code{matrix(transition,nc,nc)}, where \code{nc} is the number of components in state vector. If \code{NULL}, then estimated.
  }
  \item{measurement}{
    Measurement vector \eqn{w}. If \code{NULL}, then estimated.
  }
  \item{initial}{
    Can be either character or a vector of initial states. If it is character, then it can be \code{"optimal"}, meaning that the initial states are optimised, or \code{"backcasting"}, meaning that the initials are produced using backcasting procedure.
  }
  \item{cfType}{
    Type of Cost Function used in optimization. \code{cfType} can be: \code{MSE} (Mean Squared Error), \code{MAE} (Mean Absolute Error), \code{HAM} (Half Absolute Moment), \code{MLSTFE} - Mean Log Squared Trace Forecast Error, \code{MSTFE} - Mean Squared Trace Forecast Error and \code{MSEh} - optimisation using only h-steps ahead error. If \code{cfType!="MSE"}, then likelihood and model selection is done based on equivalent \code{MSE}. Model selection in this cases becomes not optimal.

    There are also available analytical approximations for multistep functions: \code{aMSEh}, \code{aMSTFE} and \code{aMLSTFE}. These can be useful in cases of small samples.
  }
  \item{h}{
    The forecasting horizon.
  }
  \item{holdout}{
    If \code{TRUE}, holdout of size \code{h} is taken from the end of the data.
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
    Defines type of intermittent model used. Can be: 1. \code{none}, meaning that the data should be considered as non-intermittent; 2. \code{fixed}, taking into account constant Bernoulli distribution of demand occurancies; 3. \code{croston}, based on Croston, 1972 method with SBA correction; 4. \code{tsb}, based on Teunter et al., 2011 method. 5. \code{auto} - automatic selection of intermittency type based on information criteria. The first letter can be used instead.
  }
  \item{bounds}{
    What type of bounds to use for smoothing parameters ("admissible" or "usual"). The first letter can be used instead of the whole word.
  }
  \item{silent}{
    If \code{silent="none"}, then nothing is silent, everything is printed out and drawn. \code{silent="all"} means that nothing is produced or drawn (except for warnings). In case of \code{silent="graph"}, no graph is produced. If \code{silent="legend"}, then legend of the graph is skipped. And finally \code{silent="output"} means that nothing is printed out in the console, but the graph is produced. \code{silent} also accepts \code{TRUE} and \code{FALSE}. In this case \code{silent=TRUE} is equivalent to \code{silent="all"}, while \code{silent=FALSE} is equivalent to \code{silent="none"}. The parameter also accepts first letter of words ("n", "a", "g", "l", "o").
  }
  \item{xreg}{
    Vector (either numeric or time series) or matrix (or data.frame) of exogenous variables that should be included in the model. If matrix included than columns should contain variables and rows - observations. Note that \code{xreg} should have number of observations equal either to in-sample or to the whole series. If the number of observations in \code{xreg} is equal to in-sample, then values for the holdout sample are produced using Naive.
  }
  \item{initialX}{
    Vector of initial parameters for exogenous variables. Ignored if \code{xreg} is NULL.
  }
  \item{updateX}{
    If \code{TRUE}, transition matrix for exogenous variables is estimated, introducing non-linear interractions between parameters. Prerequisite - non-NULL \code{xreg}.
  }
  \item{persistenceX}{
    Persistence vector \eqn{g_X}, containing smoothing parameters for exogenous variables. If \code{NULL}, then estimated. Prerequisite - non-NULL \code{xreg}.
  }
  \item{transitionX}{
    Transition matrix \eqn{F_x} for exogenous variables. Can be provided as a vector. Matrix will be formed using the default \code{matrix(transition,nc,nc)}, where \code{nc} is number of components in state vector. If \code{NULL}, then estimated. Prerequisite - non-NULL \code{xreg}.
  }
  \item{...}{
    Other non-documented parameters.
    For example parameter \code{model} can accept a previously estimated GES model and use all its parameters.
    \code{FI=TRUE} will make the function produce Fisher Information matrix, which then can be used to calculated variances of parameters of the model.
  }
}
\details{
    The function estimates the Single Source of Error State-space model of the following type:

    \eqn{y_[t] = o_[t] (w' v_[t-l] + x_t a_[t-1] + \epsilon_[t])}

    \eqn{v_[t] = F v_[t-l] + g \epsilon_[t]}

    \eqn{a_[t] = F_[X] a_[t-1] + g_[X] \epsilon_[t] / x_[t]}

    Where \eqn{o_[t]} is Bernoulli distributed random variable (in case of normal data equal to 1), \eqn{v_[t]} is a state vector (defined using \code{orders}) and \eqn{l} is a vector of \code{lags}, \eqn{x_t} vector of exogenous parameters.
}
\value{
  Object of class "smooth" is returned. It contains:

  \itemize{
    \item \code{model} - name of the estimated model.
    \item \code{timeElapsed} - time elapsed for the construction of the model.
    \item \code{states} - matrix of fuzzy components of GES, where \code{rows} correspond to time and \code{cols} to states.
    \item \code{initialType} - Typetof initial values used.
    \item \code{initial} - initial values of state vector (extracted from \code{states}).
    \item \code{nParam} - number of estimated parameters.
    \item \code{measurement} - matrix w.
    \item \code{transition} - matrix F.
    \item \code{persistence} - persistence vector. This is the place, where smoothing parameters live.
    \item \code{fitted} - fitted values of ETS.
    \item \code{forecast} - point forecast of ETS.
    \item \code{lower} - lower bound of prediction interval. When \code{intervals="none"} then NA is returned.
    \item \code{upper} - higher bound of prediction interval. When \code{intervals="none"} then NA is returned.
    \item \code{residuals} - the residuals of the estimated model.
    \item \code{errors} - matrix of 1 to h steps ahead errors.
    \item \code{s2} - variance of the residuals (taking degrees of freedom into account).
    \item \code{intervals} - type of intervals asked by user.
    \item \code{level} - confidence level for intervals.
    \item \code{actuals} - original data.
    \item \code{holdout} - holdout part of the original data.
    \item \code{iprob} - fitted and forecasted values of the probability of demand occurrence.
    \item \code{intermittent} - type of intermittent model fitted to the data.
    \item \code{xreg} - provided vector or matrix of exogenous variables.
    \item \code{updateX} - boolean, defining, if the states of exogenous variables were estimated as well.
    \item \code{initialX} - initial values for parameters of exogenous variables.
    \item \code{persistenceX} - persistence vector g for exogenous variables.
    \item \code{transitionX} - transition matrix F for exogenous variables.
    \item \code{ICs} - values of information criteria of the model. Includes AIC, AICc and BIC.
    \item \code{cf} - Cost function value.
    \item \code{cfType} - Type of cost function used in the estimation.
    \item \code{FI} - Fisher Information. Equal to NULL if \code{FI=FALSE} or when \code{FI} is not provided at all.
    \item \code{accuracy} - vector or accuracy measures for the holdout sample. Includes MPE, MAPE, SMAPE, MASE, MAE/mean, RelMAE and Bias coefficient (based on complex numbers). Available only when \code{holdout=TRUE}.
  }
}
\references{
No references yet.
}
\author{
Ivan Svetunkov
}

\seealso{
\code{\link[forecast]{ets}, \link[smooth]{es}, \link[smooth]{ces}, \link[smooth]{sim.es}}
}

\examples{
# Something simple:
ges(rnorm(118,100,3),orders=c(1),lags=c(1),h=18,holdout=TRUE,bounds="a",intervals="p")

# A more complicated model with seasonality
\dontrun{ourModel <- ges(rnorm(118,100,3),orders=c(2,1),lags=c(1,4),h=18,holdout=TRUE)}

# Redo previous model on a new data and produce prediction intervals
\dontrun{ges(rnorm(118,100,3),model=ourModel,h=18,intervals="sp")}

# Produce something crazy with optimal initials (not recommended)
\dontrun{ges(rnorm(118,100,3),orders=c(1,1,1),lags=c(1,3,5),h=18,holdout=TRUE,initial="o")}

# Simpler model estiamted using trace forecast error cost function and its analytical analogue
\dontrun{ges(rnorm(118,100,3),orders=c(1),lags=c(1),h=18,holdout=TRUE,bounds="n",cfType="MSTFE")
ges(rnorm(118,100,3),orders=c(1),lags=c(1),h=18,holdout=TRUE,bounds="n",cfType="aMSTFE")}

# Introduce exogenous variables
\dontrun{ges(rnorm(118,100,3),orders=c(1),lags=c(1),h=18,holdout=TRUE,xreg=c(1:118))}

# Ask for their update
\dontrun{ges(rnorm(118,100,3),orders=c(1),lags=c(1),h=18,holdout=TRUE,xreg=c(1:118),updateX=TRUE)}

# Do the same but now let's shrink parameters...
\dontrun{ges(rnorm(118,100,3),orders=c(1),lags=c(1),h=18,xreg=c(1:118),updateX=TRUE,cfType="MSTFE")

ourModel <- ges(rnorm(118,100,3),orders=c(1),lags=c(1),h=18,holdout=TRUE,cfType="aMSTFE")

summary(ourModel)
forecast(ourModel)
plot(forecast(ourModel))}
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{ ges }
\keyword{ Exponential Smoothing }% __ONLY ONE__ keyword per line
