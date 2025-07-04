% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/adam.R, R/autoadam.R, R/sm.R
\name{adam}
\alias{adam}
\alias{simulate.adam}
\alias{auto.adam}
\alias{sm.adam}
\title{ADAM is Augmented Dynamic Adaptive Model}
\usage{
adam(data, model = "ZXZ", lags = c(frequency(data)), orders = list(ar =
  c(0), i = c(0), ma = c(0), select = FALSE), constant = FALSE,
  formula = NULL, regressors = c("use", "select", "adapt"),
  occurrence = c("none", "auto", "fixed", "general", "odds-ratio",
  "inverse-odds-ratio", "direct"), distribution = c("default", "dnorm",
  "dlaplace", "ds", "dgnorm", "dlnorm", "dinvgauss", "dgamma"),
  loss = c("likelihood", "MSE", "MAE", "HAM", "LASSO", "RIDGE", "MSEh",
  "TMSE", "GTMSE", "MSCE"), outliers = c("ignore", "use", "select"),
  level = 0.99, h = 0, holdout = FALSE, persistence = NULL,
  phi = NULL, initial = c("backcasting", "optimal", "two-stage",
  "complete"), arma = NULL, ic = c("AICc", "AIC", "BIC", "BICc"),
  bounds = c("usual", "admissible", "none"), silent = TRUE,
  ets = c("conventional", "adam"), ...)

\method{simulate}{adam}(object, nsim = 1, seed = NULL,
  obs = nobs(object), ...)

auto.adam(data, model = "ZXZ", lags = c(frequency(data)),
  orders = list(ar = c(3, 3), i = c(2, 1), ma = c(3, 3), select = TRUE),
  formula = NULL, regressors = c("use", "select", "adapt"),
  occurrence = c("none", "auto", "fixed", "general", "odds-ratio",
  "inverse-odds-ratio", "direct"), distribution = c("dnorm", "dlaplace",
  "ds", "dgnorm", "dlnorm", "dinvgauss", "dgamma"), outliers = c("ignore",
  "use", "select"), level = 0.99, h = 0, holdout = FALSE,
  persistence = NULL, phi = NULL, initial = c("backcasting", "optimal",
  "two-stage", "complete"), arma = NULL, ic = c("AICc", "AIC", "BIC",
  "BICc"), bounds = c("usual", "admissible", "none"), silent = TRUE,
  parallel = FALSE, ets = c("conventional", "adam"), ...)

\method{sm}{adam}(object, model = "YYY", lags = NULL, orders = list(ar =
  c(0), i = c(0), ma = c(0), select = FALSE), constant = FALSE,
  formula = NULL, regressors = c("use", "select", "adapt"), data = NULL,
  persistence = NULL, phi = NULL, initial = c("optimal", "backcasting"),
  arma = NULL, ic = c("AICc", "AIC", "BIC", "BICc"), bounds = c("usual",
  "admissible", "none"), silent = TRUE, ...)
}
\arguments{
\item{data}{Vector, containing data needed to be forecasted. If a matrix (or
data.frame / data.table) is provided, then the first column is used as a
response variable, while the rest of the matrix is used as a set of explanatory
variables. \code{formula} can be used in the latter case in order to define what
relation to have.}

\item{model}{The type of ETS model. The first letter stands for the type of
the error term ("A" or "M"), the second (and sometimes the third as well) is for
the trend ("N", "A", "Ad", "M" or "Md"), and the last one is for the type of
seasonality ("N", "A" or "M"). In case of several lags, the seasonal components
are assumed to be the same. The model is then printed out as
ETS(M,Ad,M)[m1,m2,...], where m1, m2, ... are the lags specified by the
\code{lags} parameter.
There are several options for the \code{model} besides the conventional ones,
which rely on information criteria:
\enumerate{
\item \code{model="ZZZ"} means that the model will be selected based on the
chosen information criteria type. The Branch and Bound is used in the process.
\item \code{model="XXX"} means that only additive components are tested, using
Branch and Bound.
\item \code{model="YYY"} implies selecting between multiplicative components.
\item \code{model="CCC"} triggers the combination of forecasts of models using
information criteria weights (Kolassa, 2011).
\item combinations between these four and the classical components are also
accepted. For example, \code{model="CAY"} will combine models with additive
trend and either none or multiplicative seasonality.
\item \code{model="PPP"} will produce the selection between pure additive and
pure multiplicative models. "P" stands for "Pure". This cannot be mixed with
other types of components.
\item \code{model="FFF"} will select between all the 30 types of models. "F"
stands for "Full". This cannot be mixed with other types of components.
\item \code{model="SSS"} creates a pool of 19 standard sensible models, which
have finite variance. This can be combined with "X" or "Y" to further restrict
the pool. For example, \code{model="SXS"} corresponds to the default pool of 15
models in the \code{ets()} function from the forecast package.
\item The parameter \code{model} can also be a vector of names of models for a
finer tuning (pool of models). For example, \code{model=c("ANN","AAA")} will
estimate only two models and select the best of them.
}

Also, \code{model} can accept a previously estimated adam and use all
its parameters.

Keep in mind that model selection with "Z" components uses Branch and Bound
algorithm and may skip some models that could have slightly smaller
information criteria. If you want to do a exhaustive search, you would need
to list all the models to check as a vector.

The default value is set to \code{"ZXZ"}, because the multiplicative trend is explosive
and dangerous. It should be used only for each separate time series, not for the
automated predictions for big  datasets.}

\item{lags}{Defines lags for the corresponding components. All components
count, starting from level, so ETS(M,M,M) model for monthly data will have
\code{lags=c(1,1,12)}. However, the function will also accept \code{lags=c(12)},
assuming that the lags 1 were dropped. In case of ARIMA, lags specify what should be
the seasonal component lag. e.g. \code{lags=c(1,12)} will lead to the
seasonal ARIMA with m=12. This can accept several lags, supporting multiple seasonal ETS
and ARIMA models.}

\item{orders}{The order of ARIMA to be included in the model. This should be passed
either as a vector (in which case the non-seasonal ARIMA is assumed) or as a list of
a type \code{orders=list(ar=c(p,P),i=c(d,D),ma=c(q,Q))}, in which case the \code{lags}
variable is used in order to determine the seasonality m. See \link[smooth]{msarima}
for details.
In addition, \code{orders} accepts one more parameter: \code{orders=list(select=FALSE)}.
If \code{TRUE}, then the function will select the most appropriate order using a
mechanism similar to \code{auto.msarima()}, but implemented in \code{auto.adam()}.
The values \code{list(ar=...,i=...,ma=...)} specify the maximum orders to check in
this case.}

\item{constant}{Logical, determining, whether the constant is needed in the model or not.
This is mainly needed for ARIMA part of the model, but can be used for ETS as well. In
case of pure regression, this is completely ignored (use \code{formula} instead).}

\item{formula}{Formula to use in case of explanatory variables. If \code{NULL},
then all the variables are used as is. Can also include \code{trend}, which would add
the global trend. Only needed if \code{data} is a matrix or if \code{trend} is provided.}

\item{regressors}{The variable defines what to do with the provided explanatory
variables:
\code{"use"} means that all of the data should be used, while
\code{"select"} means that a selection using \code{ic} should be done,
\code{"adapt"} will trigger the mechanism of time varying parameters for the
explanatory variables.}

\item{occurrence}{The type of model used in probability estimation. Can be
\code{"none"} - none,
\code{"fixed"} - constant probability,
\code{"general"} - the general Beta model with two parameters,
\code{"odds-ratio"} - the Odds-ratio model with b=1 in Beta distribution,
\code{"inverse-odds-ratio"} - the model with a=1 in Beta distribution,
\code{"direct"} - the TSB-like (Teunter et al., 2011) probability update
mechanism a+b=1,
\code{"auto"} - the automatically selected type of occurrence model.

The type of model used in the occurrence is equal to the one provided in the
\code{model} parameter.

Also, a model produced using \link[smooth]{oes} or \link[greybox]{alm} function
can be used here.}

\item{distribution}{what density function to assume for the error term. The full
name of the distribution should be provided, starting with the letter "d" -
"density". The names align with the names of distribution functions in R.
For example, see \link[stats]{dnorm}. For detailed explanation of available
distributions, see vignette in greybox package: \code{vignette("greybox","alm")}.}

\item{loss}{The type of Loss Function used in optimization. \code{loss} can
be:
\itemize{
\item \code{likelihood} - the model is estimated via the maximisation of the
likelihood of the function specified in \code{distribution};
\item \code{MSE} (Mean Squared Error),
\item \code{MAE} (Mean Absolute Error),
\item \code{HAM} (Half Absolute Moment),
\item \code{LASSO} - use LASSO to shrink the parameters of the model;
\item \code{RIDGE} - use RIDGE to shrink the parameters of the model;
\item \code{TMSE} - Trace Mean Squared Error,
\item \code{GTMSE} - Geometric Trace Mean Squared Error,
\item \code{MSEh} - optimisation using only h-steps ahead error,
\item \code{MSCE} - Mean Squared Cumulative Error.
}
In case of LASSO / RIDGE, the variables are not normalised prior to the estimation,
but the parameters are divided by the mean values of explanatory variables.

Note that model selection and combination works properly only for the default
\code{loss="likelihood"}.

Furthermore, just for fun the absolute and half analogues of multistep estimators
are available: \code{MAEh}, \code{TMAE}, \code{GTMAE}, \code{MACE},
\code{HAMh}, \code{THAM}, \code{GTHAM}, \code{CHAM}.

Last but not least, user can provide their own function here as well, making sure
that it accepts parameters \code{actual}, \code{fitted} and \code{B}. Here is an
example:

\code{lossFunction <- function(actual, fitted, B) return(mean(abs(actual-fitted)))}

\code{loss=lossFunction}}

\item{outliers}{Defines what to do with outliers: \code{"ignore"}, so just returning the model,
\code{"use"} - detect outliers based on specified \code{level} and include dummies for them in the model,
or detect and \code{"select"} those of them that reduce \code{ic} value.}

\item{level}{What confidence level to use for detection of outliers. The default is 99\%. The specific
bounds of confidence interval depend on the distribution used in the model.}

\item{h}{The forecast horizon. Mainly needed for the multistep loss functions.}

\item{holdout}{Logical. If \code{TRUE}, then the holdout of the size \code{h}
is taken from the data (can be used for the model testing purposes).}

\item{persistence}{Persistence vector \eqn{g}, containing smoothing
parameters. If \code{NULL}, then estimated. Can be also passed as a names list of
the type: \code{persistence=list(level=0.1, trend=0.05, seasonal=c(0.1,0.2),
xreg=c(0.1,0.2))}. Dropping some elements from the named list will make the function
estimate them. e.g. if you don't specify seasonal in the persistence for the ETS(M,N,M)
model, it will be estimated.}

\item{phi}{Value of damping parameter. If \code{NULL} then it is estimated.
Only applicable for damped-trend models.}

\item{initial}{Can be either character or a list, or a vector of initial states.
If it is character, then it can be \code{"backcasting"}, meaning that the initials of
dynamic part of the model are produced using backcasting procedure (advised
for data with high frequency), or \code{"optimal"}, meaning that all initial
states are optimised, or \code{"two-stage"}, meaning that optimisation is done
after the backcasting, refining the states. In case of backcasting, the parameters of the
explanatory variables are optimised. This is recommended for ETSX and ARIMAX
models. Alternatively, you can set \code{initial="complete"} backcasting,
which means that all states (including explanatory variables) are initialised
via backcasting.

If a use provides a list of values, it is recommended to use the named one and
to provide the initial components that are available. For example:
\code{initial=list(level=1000,trend=10,seasonal=list(c(1,2),c(1,2,3,4)),
arima=1,xreg=100)}. If some of the components are needed by the model, but are
not provided in the list, they will be estimated. If the vector is provided,
then it is expected that the components will be provided in the same order as above,
one after another without any gaps.}

\item{arma}{Either the named list or a vector with AR / MA parameters ordered lag-wise.
The number of elements should correspond to the specified orders e.g.
\code{orders=list(ar=c(1,1),ma=c(1,1)), lags=c(1,4), arma=list(ar=c(0.9,0.8),ma=c(-0.3,0.3))}}

\item{ic}{The information criterion to use in the model selection / combination
procedure.}

\item{bounds}{The type of bounds for the persistence to use in the model
estimation. Can be either \code{admissible} - guaranteeing the stability of the
model, \code{usual} - restricting the values with (0, 1) or \code{none} - no
restrictions (potentially dangerous).}

\item{silent}{Specifies, whether to provide the progress of the function or not.
If \code{TRUE}, then the function will print what it does and how much it has
already done.}

\item{ets}{Parameter determining, which ETS formulation to use. If \code{ets="conventional"},
the one from Hyndman et al. (2008) is used. In case of \code{ets="adam"}, ADAM reformulation
that updates multiplicative components differently is used. The latter is closer
to applying ETS to log-transformed data when multiplicative components are used. This helps
makig trend less explosive.}

\item{...}{Other non-documented parameters. For example, \code{FI=TRUE} will
make the function also produce Fisher Information matrix, which then can be
used to calculated variances of smoothing parameters and initial states of
the model. This is calculated based on the hessian of log-likelihood function and
accepts \code{stepSize} parameter, determining how it is calculated. The default value
is \code{stepSize=.Machine$double.eps^(1/4)}. This is used in the \link[stats]{vcov} method.
Number of iterations inside the backcasting loop to do is regulated with \code{nIterations}
parameter. By default it is set to 2. Furthermore, starting values of parameters can be
passed via \code{B}, while the upper and lower bounds should be passed in \code{ub}
and \code{lb} respectively. In this case they will be used for optimisation. These
values should have the length equal to the number of parameters to estimate in
the following order:
\enumerate{
\item All smoothing parameters (for the states and then for the explanatory variables);
\item Damping parameter (if needed);
\item ARMA parameters;
\item All the initial values (for the states and then for the explanatory variables).
}
You can also pass parameters to the optimiser in order to fine tune its work:
\itemize{
\item \code{maxeval} - maximum number of evaluations to carry out. The default is 40 per
estimated parameter for ETS and / or ARIMA and at least 1000 if explanatory variables
are introduced in the model (100 per parameter for explanatory variables, but not less
than 1000);
\item \code{maxtime} - stop, when the optimisation time (in seconds) exceeds this;
\item \code{xtol_rel} - the relative precision of the optimiser (the default is 1E-6);
\item \code{xtol_abs} - the absolute precision of the optimiser (the default is 1E-8);
\item \code{ftol_rel} - the stopping criterion in case of the relative change in the loss
function (the default is 1E-8);
\item \code{ftol_abs} - the stopping criterion in case of the absolute change in the loss
function (the default is 0 - not used);
\item \code{algorithm} - the algorithm to use in optimisation
(by default, \code{"NLOPT_LN_NELDERMEAD"} is used);
\item \code{print_level} - the level of output for the optimiser (0 by default).
If equal to 41, then the detailed results of the optimisation are returned.
}
You can read more about these parameters by running the function
\link[nloptr]{nloptr.print.options}.
It is also possible to regulate what smoother to use to get initial seasonal indices
from the \link[smooth]{msdecompose} function via the \code{smoother} parameter.
Finally, the parameter \code{lambda} for LASSO / RIDGE, \code{alpha} for the Asymmetric
Laplace, \code{shape} for the Generalised Normal and \code{nu} for Student's distributions
can be provided here as well.}

\item{object}{The model previously estimated using \code{adam()} function.}

\item{nsim}{Number of series to generate from the model.}

\item{seed}{Random seed used in simulation of data.}

\item{obs}{Number of observations to produce in the simulated data.}

\item{parallel}{If TRUE, the estimation of ADAM models is done in parallel (used in \code{auto.adam} only).
If the number is provided (e.g. \code{parallel=41}), then the specified number of cores is set up.
WARNING! Packages \code{foreach} and either \code{doMC} (Linux and Mac only)
or \code{doParallel} are needed in order to run the function in parallel.}
}
\value{
Object of class "adam" is returned. It contains the list of the
following values:
\itemize{
\item \code{model} - the name of the constructed model,
\item \code{timeElapsed} - the time elapsed for the estimation of the model,
\item \code{data} - the in-sample part of the data used for the training of the model. Includes
the actual values in the first column,
\item \code{holdout} - the holdout part of the data, excluded for purposes of model evaluation,
\item \code{fitted} - the vector of fitted values,
\item \code{residuals} - the vector of residuals,
\item \code{forecast} - the point forecast for h steps ahead (by default NA is returned). NOTE
that these do not always correspond to the conditional expectations for ETS models. See ADAM
textbook, Section 6.4. for details (\url{https://openforecast.org/adam/ETSTaxonomyMaths.html}),
\item \code{states} - the matrix of states with observations in rows and states in columns,
\item \code{persisten} - the vector of smoothing parameters,
\item \code{phi} - the value of damping parameter,
\item \code{transition} - the transition matrix,
\item \code{measurement} - the measurement matrix with observations in rows and state elements
in columns,
\item \code{initial} - the named list of initial values, including level, trend, seasonal, ARIMA
and xreg components,
\item \code{initialEstimated} - the named vector, defining which of the initials were estimated in
the model,
\item \code{initialType} - the type of initialisation used (backcasting/optimal/two-stage/complete/provided),
\item \code{orders} - the orders of ARIMA used in the estimation,
\item \code{constant} - the value of the constant (if it was included),
\item \code{arma} - the list of AR / MA parameters used in the model,
\item \code{nParam} - the matrix of the estimated / provided parameters,
\item \code{occurrence} - the oes model used for the occurrence part of the model,
\item \code{formula} - the formula used for the explanatory variables expansion,
\item \code{loss} - the type of loss function used in the estimation,
\item \code{lossValue} - the value of that loss function,
\item \code{logLik} - the value of the log-likelihood,
\item \code{distribution} - the distribution function used in the calculation of the likelihood,
\item \code{scale} - the value of the scale parameter,
\item \code{lambda} - the value of the parameter used in LASSO / dalaplace / dt,
\item \code{B} - the vector of all estimated parameters,
\item \code{lags} - the vector of lags used in the model construction,
\item \code{lagsAll} - the vector of the internal lags used in the model,
\item \code{profile} - the matrix with the profile used in the construction of the model,
\item \code{profileInitial} - the matrix with the initial profile (for the before the sample values),
\item \code{call} - the call used in the evaluation,
\item \code{bounds} - the type of bounds used in the process,
\item \code{res} - result of the model estimation, the output of the \code{nloptr()} function, explaining
how optimisation went,
\item \code{other} - the list with other parameters, such as shape for distributions or ARIMA
polynomials.
}
}
\description{
Function constructs an advanced Single Source of Error model, based on ETS
taxonomy and ARIMA elements
}
\details{
Function estimates ADAM in a form of the Single Source of Error state space
model of the following type:

\deqn{y_{t} = o_t (w(v_{t-l}) + h(x_t, a_{t-1}) + r(v_{t-l}) \epsilon_{t})}

\deqn{v_{t} = f(v_{t-l}, a_{t-1}) + g(v_{t-l}, a_{t-1}, x_{t}) \epsilon_{t}}

Where \eqn{o_{t}} is the Bernoulli distributed random variable (in case of
normal data it equals to 1 for all observations), \eqn{v_{t}} is the state
vector and \eqn{l} is the vector of lags, \eqn{x_t} is the vector of
exogenous variables. w(.) is the measurement function, r(.) is the error
function, f(.) is the transition function, g(.) is the persistence
function and \eqn{a_t} is the vector of parameters for exogenous variables.
Finally, \eqn{\epsilon_{t}} is the error term.

The implemented model allows introducing several seasonal states and supports
intermittent data via the \code{occurrence} variable.

The error term \eqn{\epsilon_t} can follow different distributions, which
are regulated via the \code{distribution} parameter. This includes:
\enumerate{
\item \code{default} - Normal distribution is used for the Additive error models,
Gamma is used for the Multiplicative error models.
\item dnorm - \link[stats]{Normal} distribution,
\item \link[greybox]{dlaplace} - Laplace distribution,
\item \link[greybox]{ds} - S distribution,
\item \link[greybox]{dgnorm} - Generalised Normal distribution,
\item \link[stats]{dlnorm} - Log-Normal distribution,
\item \link[stats]{dgamma} - Gamma distribution,
\item \link[statmod]{dinvgauss} - Inverse Gaussian distribution,
}

For some more information about the model and its implementation, see the
vignette: \code{vignette("adam","smooth")}. The more detailed explanation
of ADAM is provided by Svetunkov (2021).

The function \code{auto.adam()} tries out models with the specified
distributions and returns the one with the most suitable one based on selected
information criterion.

\link[greybox]{sm}.adam method estimates the scale model for the already
estimated adam. In order for ADAM to take the SM model into account, the
latter needs to be recorded in the former, amending the likelihood and the number
of degrees of freedom. This can be done using \link[greybox]{implant} method.
}
\examples{

### The main examples are provided in the adam vignette, check it out via:
\dontrun{vignette("adam","smooth")}

# Model selection using a specified pool of models
ourModel <- adam(rnorm(100,100,10), model=c("ANN","ANA","AAA"), lags=c(5,10))
\donttest{adamSummary <- summary(ourModel)
xtable(adamSummary)}

\donttest{forecast(ourModel)
par(mfcol=c(3,4))
plot(ourModel, c(1:11))}

# Model combination using a specified pool
\donttest{ourModel <- adam(rnorm(100,100,10), model=c("ANN","AAN","MNN","CCC"),
                          lags=c(5,10))}

# ADAM ARIMA
\donttest{ourModel <- adam(rnorm(100,100,10), model="NNN",
                          lags=c(1,4), orders=list(ar=c(1,0),i=c(1,0),ma=c(1,1)))}

# Fit ADAM to the data
ourModel <- adam(rnorm(100,100,10), model="AAdN")
# Simulate the data
x <- simulate(ourModel)

# Automatic selection of appropriate distribution and orders of ADAM ETS+ARIMA
\donttest{ourModel <- auto.adam(rnorm(100,100,10), model="ZZN", lags=c(1,4),
                      orders=list(ar=c(2,2),ma=c(2,2),select=TRUE))}

}
\references{
\itemize{
\item Svetunkov I. (2023) Smooth forecasting with the smooth package in R. arXiv:2301.01790.
\doi{10.48550/arXiv.2301.01790}.
\item Svetunkov I. (2015 - Inf) "smooth" package for R - series of posts about the underlying
models and how to use them: \url{https://openforecast.org/category/r-en/smooth/}.
}

\itemize{
\item Svetunkov, I. (2023). Forecasting and Analytics with the Augmented
Dynamic Adaptive Model (ADAM) (1st ed.). Chapman and Hall/CRC.
\doi{10.1201/9781003452652}, online version: \url{https://openforecast.org/adam/}.
}

\itemize{
\item Svetunkov, I., 2023. Smooth Forecasting with the Smooth Package in R. arXiv.
\doi{10.48550/arXiv.2301.01790}
\item Snyder, R. D., 1985. Recursive Estimation of Dynamic Linear Models.
Journal of the Royal Statistical Society, Series B (Methodological) 47 (2), 272-276.
\item Hyndman, R.J., Koehler, A.B., Ord, J.K., and Snyder, R.D. (2008)
Forecasting with exponential smoothing: the state space approach,
Springer-Verlag. \doi{10.1007/978-3-540-71918-2}.
}

\itemize{
\item Svetunkov, I., Boylan, J.E., 2023a. iETS: State Space Model for
Intermittent Demand Forecastings. International Journal of Production
Economics. 109013. \doi{10.1016/j.ijpe.2023.109013}
\item Teunter R., Syntetos A., Babai Z. (2011). Intermittent demand:
Linking forecasting to inventory obsolescence. European Journal of
Operational Research, 214, 606-615.
\item Croston, J. (1972) Forecasting and stock control for intermittent
demands. Operational Research Quarterly, 23(3), 289-303.
}

\itemize{
\item Kolassa, S. (2011) Combining exponential smoothing forecasts using Akaike
weights. International Journal of Forecasting, 27, pp 238 - 251.
\item Svetunkov, I., Boylan, J.E., 2023b. Staying Positive: Challenges and
 Solutions in Using Pure Multiplicative ETS Models. IMA Journal of
 Management Mathematics. p. 403-425. \doi{10.1093/imaman/dpad028}
}

\itemize{
\item Taylor, J.W. and Bunn, D.W. (1999) A Quantile Regression Approach to
Generating Prediction Intervals. Management Science, Vol 45, No 2, pp
225-237.
\item Lichtendahl Kenneth C., Jr., Grushka-Cockayne Yael, Winkler
Robert L., (2013) Is It Better to Average Probabilities or
Quantiles? Management Science 59(7):1594-1611. DOI:
\doi{10.1287/mnsc.1120.1667}
}
}
\seealso{
\code{\link[smooth]{es}, \link[smooth]{msarima}}
}
\author{
Ivan Svetunkov, \email{ivan@svetunkov.com}
}
\keyword{models}
\keyword{nonlinear}
\keyword{regression}
\keyword{smooth}
\keyword{ts}
\keyword{univar}
