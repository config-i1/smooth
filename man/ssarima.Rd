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
ssarima(data, orders=list(ar=0,i=c(1),ma=c(1)), lags=c(1),
  constant=FALSE, AR=NULL, MA=NULL,
  initial=c("backcasting","optimal"),
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
    Data that needs to be forecasted.
  }
  \item{orders}{
    List of orders, containing vector variables \code{ar}, \code{i} and \code{ma}. Example: \code{orders=list(ar=c(1,2),i=c(1),ma=c(1,1,1))}. If a variable is not provided in the list, then it is assumed to be equal to zero. At least one variable should have the same length as \code{lags}.
  }
  \item{lags}{
    Defines lags for the corresponding orders (see examples above). The length of \code{lags} must correspond to the length of either \code{ar}, \code{i} or \code{ma} in \code{orders} variable. There is no restrictions on the length of \code{lags} vector. It is recommended to order \code{lags} ascending.
  }
  \item{constant}{
    If \code{TRUE}, constant term is included in the model. Can also be a number (constant value).
  }
  \item{AR}{
    Vector or matrix of AR parameters. The order of parameters should be lag-wise. This means that first all the AR parameters of the firs lag should be passed, then for the second etc. AR of another ssarima can be passed here.
  }
  \item{MA}{
    Vector or matrix of MA parameters. The order of parameters should be lag-wise. This means that first all the MA parameters of the firs lag should be passed, then for the second etc. MA of another ssarima can be passed here.
  }
  \item{initial}{
    Can be either character or a vector of initial states. If it is character, then it can be \code{"optimal"}, meaning that the initial states are optimised, or \code{"backcasting"}, meaning that the initials are produced using backcasting procedure (advised for data with high frequency). If character, then \code{initial.season} will be estimated in the way defined by \code{initial}.
  }
  \item{cfType}{
    Type of Cost Function used in optimization. \code{cfType} can be: \code{MSE} (Mean Squared Error), \code{MAE} (Mean Absolute Error), \code{HAM} (Half Absolute Moment), \code{MLSTFE} - Mean Log Squared Trace Forecast Error, \code{MSTFE} - Mean Squared Trace Forecast Error and \code{MSEh} - optimisation using only h-steps ahead error. If \code{cfType!="MSE"}, then likelihood and model selection is done based on equivalent \code{MSE}. Model selection in this cases becomes not optimal.

    There are also available analytical approximations for multistep functions: \code{aMSEh}, \code{aMSTFE} and \code{aMLSTFE}. These can be useful in cases of small samples.
  }
  \item{h}{ the forecasting horizon.
  }
  \item{holdout}{ if \code{TRUE}, the holdout of the size \code{h} is taken from the end of the data.
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
    What type of bounds to use for smoothing parameters ("admissible" or "none"). The first letter can be used instead of the whole word.
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

    Vectors of orders can be passed here using \code{ar.orders}, \code{i.orders} and \code{ma.orders}. \code{orders} variable needs to be NULL in this case.

    Parameter \code{model} can accept a previously estimated SSARIMA model and use all its parameters.

    \code{FI=TRUE} will make the function produce Fisher Information matrix, which then can be used to calculated variances of parameters of the model.
  }
}
\details{
The basic ARIMA(p,d,q) used in the function has the following form:
\eqn{(1 - B)^d (1 - a_1 B - a_2 B^2 - ... - a_p B^p) y_[t] = (1 - b_1 B - b_2 B^2 - ... - b_q B^q) \epsilon_[t] + c}
where \eqn{y_[t]} is actual values,  \eqn{\epsilon_[t]} is error term, \eqn{a_i, b_j} are parameters for AR and MA respectively and \eqn{c} is constant. In case of non-zero differences \eqn{c} starts acting as drift.

This model is then transformed into ARIMA in the Single Source of Error State-space form (proposed in Snyder, 1985):
\eqn{y_[t] = o_[t] (w' v_[t-l] + x_t a_[t-1] + \epsilon_[t])}
\eqn{v_[t] = F v_[t-1] + g \epsilon_[t]}
\eqn{a_[t] = F_[X] a_[t-1] + g_[X] \epsilon_[t] / x_[t]}
where \eqn{o_[t]} is Bernoulli distributed random variable (in case of normal data equals to 1 for all observations), \eqn{v_[t]} is a state vector (defined using \code{ar.orders} and \code{i.orders}), \eqn{x_t} vector of exogenous parameters.

Due to the flexibility of the model, multiple seasonalities can be used. For example, something crazy like this can be constructed: SARIMA(1,1,1)(0,1,1)[24](2,0,1)[24*7](0,0,1)[24*30], but the estimation may take a lot of time...
}
\value{
  Object of class "smooth" is returned. It contains the list of the following values:

  \itemize{
    \item \code{model} - the name of the estimated model.
    \item \code{timeElapsed} - time elapsed for the construction of the model.
    \item \code{states} - the matrix of the fuzzy components of ssarima, where \code{rows} correspond to time and \code{cols} to states.
    \item \code{transition} - matrix F.
    \item \code{persistence} - the persistence vector. This is the place, where smoothing parameters live.
    \item \code{AR} - the matrix of coefficients of AR terms.
    \item \code{I} - the matrix of coefficients of I terms.
    \item \code{MA} - the matrix of coefficients of MA terms.
    \item \code{constant} - the value of the constant term.
    \item \code{initialType} - Typetof initial values used.
    \item \code{initial} - the initial values of the state vector (extracted from \code{states}).
    \item \code{nParam} - number of estimated parameters.
    \item \code{fitted} - the fitted values of ETS.
    \item \code{forecast} - the point forecast of ETS.
    \item \code{lower} - the lower bound of prediction interval. When \code{intervals="none"} then NA is returned.
    \item \code{upper} - the higher bound of prediction interval. When \code{intervals="none"} then NA is returned.
    \item \code{residuals} - the residuals of the estimated model.
    \item \code{errors} - The matrix of 1 to h steps ahead errors.
    \item \code{s2} - variance of the residuals (taking degrees of freedom into account).
    \item \code{intervals} - type of intervals asked by user.
    \item \code{level} - confidence level for intervals.
    \item \code{actuals} - the original data.
    \item \code{holdout} - the holdout part of the original data.
    \item \code{iprob} - the fitted and forecasted values of the probability of demand occurrence.
    \item \code{intermittent} - type of intermittent model fitted to the data.
    \item \code{xreg} - the provided vector or matrix of exogenous variables.
    \item \code{updateX} - boolean, defining, if the states of exogenous variables were estimated as well.
    \item \code{initialX} - initial values for parameters of exogenous variables.
    \item \code{persistenceX} - persistence vector g for exogenous variables.
    \item \code{transitionX} - transition matrix F for exogenous variables.
    \item \code{ICs} - values of information criteria of the model. Includes AIC, AICc and BIC.
    \item \code{cf} - Cost function value.
    \item \code{cfType} - Type of cost function used in the estimation.
    \item \code{FI} - Fisher Information. Equal to NULL if \code{FI=FALSE} or when \code{FI} is not provided at all.
    \item \code{accuracy} - the vector or accuracy measures for the holdout sample. Includes MPE, MAPE, SMAPE, MASE, MAE/mean, RelMAE and Bias coefficient (based on complex numbers). Available only when \code{holdout=TRUE}.
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
\code{\link[forecast]{auto.arima}, \link[smooth]{orders}, \link[smooth]{lags}, \link[smooth]{sim.ssarima}}
}

\examples{
# ARIMA(1,1,1) fitted to some data
ourModel <- ssarima(rnorm(118,100,3),orders=list(ar=c(1),i=c(1),ma=c(1)),lags=c(1),h=18,
                             holdout=TRUE,intervals="p")

# The previous one is equivalent to:
\dontrun{ourModel <- ssarima(rnorm(118,100,3),ar.orders=c(1),i.orders=c(1),ma.orders=c(1),lags=c(1),h=18,
                    holdout=TRUE,intervals="p")}

# Model with the same lags and orders, applied to a different data
ssarima(rnorm(118,100,3),orders=orders(ourModel),lags=lags(ourModel),h=18,holdout=TRUE)

# The same model applied to a different data
ssarima(rnorm(118,100,3),model=ourModel,h=18,holdout=TRUE)

# Example of SARIMA(2,0,0)(1,0,0)[4]
\dontrun{ssarima(rnorm(118,100,3),orders=list(ar=c(2,1)),lags=c(1,4),h=18,holdout=TRUE)}

# SARIMA(1,1,1)(0,0,1)[4] with different initialisations
\dontrun{ssarima(rnorm(118,100,3),orders=list(ar=c(1),i=c(1),ma=c(1,1)),
        lags=c(1,4),h=18,holdout=TRUE)
ssarima(rnorm(118,100,3),orders=list(ar=c(1),i=c(1),ma=c(1,1)),
        lags=c(1,4),h=18,holdout=TRUE,initial="o")}

# SARIMA of a perculiar order on AirPassengers data
\dontrun{ssarima(AirPassengers,orders=list(ar=c(1,0,3),i=c(1,0,1),ma=c(0,1,2)),lags=c(1,6,12),
        h=10,holdout=TRUE)}

# ARIMA(1,1,1) with Mean Squared Trace Forecast Error
\dontrun{ssarima(rnorm(118,100,3),orders=list(ar=1,i=1,ma=1),lags=1,h=18,holdout=TRUE,cfType="MSTFE")
ssarima(rnorm(118,100,3),orders=list(ar=1,i=1,ma=1),lags=1,h=18,holdout=TRUE,cfType="aMSTFE")}

# SARIMA(0,1,1) with exogenous variables
ssarima(rnorm(118,100,3),orders=list(i=1,ma=1),h=18,holdout=TRUE,xreg=c(1:118))

# SARIMA(0,1,1) with exogenous variables with crazy estimation of xreg
ourModel <- ssarima(rnorm(118,100,3),orders=list(i=1,ma=1),h=18,holdout=TRUE,
                    xreg=c(1:118),updateX=TRUE)

summary(ourModel)
forecast(ourModel)
plot(forecast(ourModel))
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{ SARIMA }
\keyword{ ARIMA }% __ONLY ONE__ keyword per line
