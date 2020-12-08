utils::globalVariables(c("adamFitted","algorithm","arEstimate","arOrders","arRequired","arimaModel",
                         "arimaPolynomials","armaParameters","componentsNamesARIMA","componentsNamesETS",
                         "componentsNumberARIMA","componentsNumberETS","componentsNumberETSNonSeasonal",
                         "componentsNumberETSSeasonal","digits","etsModel","ftol_abs","ftol_rel",
                         "horizon","iOrders","iRequired","initialArima","initialArimaEstimate",
                         "initialArimaNumber","initialLevel","initialLevelEstimate","initialSeasonal",
                         "initialSeasonalEstimate","initialTrend","initialTrendEstimate","lagsModelARIMA",
                         "lagsModelAll","lagsModelSeasonal","profilesObservedTable","profilesRecentTable",
                         "other","otherParameterEstimate","lambda","lossFunction",
                         "maEstimate","maOrders","maRequired","matVt","matWt","maxtime","modelIsTrendy",
                         "nParamEstimated","persistenceLevel","persistenceLevelEstimate",
                         "persistenceSeasonal","persistenceSeasonalEstimate","persistenceTrend",
                         "persistenceTrendEstimate","vecG","xtol_abs","xtol_rel","stepSize","yClasses",
                         "yForecastIndex","yInSampleIndex","yIndexAll","yNAValues","yStart","responseName",
                         "xregParametersMissing","xregParametersIncluded","xregParametersEstimated",
                         "xregParametersPersistence","xregModelInitials"));

#' ADAM is Augmented Dynamic Adaptive Model
#'
#' Function constructs an advanced Single Source of Error model, based on ETS
#' taxonomy and ARIMA elements
#'
#' Function estimates ADAM in a form of the Single Source of Error state space
#' model of the following type:
#'
#' \deqn{y_{t} = o_t (w(v_{t-l}) + h(x_t, a_{t-1}) + r(v_{t-l}) \epsilon_{t})}
#'
#' \deqn{v_{t} = f(v_{t-l}, a_{t-1}) + g(v_{t-l}, a_{t-1}, x_{t}) \epsilon_{t}}
#'
#' Where \eqn{o_{t}} is the Bernoulli distributed random variable (in case of
#' normal data it equals to 1 for all observations), \eqn{v_{t}} is the state
#' vector and \eqn{l} is the vector of lags, \eqn{x_t} is the vector of
#' exogenous variables. w(.) is the measurement function, r(.) is the error
#' function, f(.) is the transition function, g(.) is the persistence
#' function and \eqn{a_t} is the vector of parameters for exogenous variables.
#' Finally, \eqn{\epsilon_{t}} is the error term.
#'
#' The implemented model allows introducing several seasonal states and supports
#' intermittent data via the \code{occurrence} variable.
#'
#' The error term \eqn{\epsilon_t} can follow different distributions, which
#' are regulated via the \code{distribution} parameter. This includes:
#' \enumerate{
#' \item \code{default} - Normal distribution is used for the Additive error models,
#' Inverse Gaussian is used for the Multiplicative error models.
#' \item \link[stats]{Normal} - Normal distribution,
#' \item \link[greybox]{dlaplace} - Laplace distribution,
#' \item \link[greybox]{ds} - S distribution,
#' \item dgnorm - Generalised Normal distribution,
# \item \link[stats]{dlogis} - Logistic Distribution,
# \item \link[stats]{dt} - T distribution,
#' \item \link[greybox]{dalaplace} - Asymmetric Laplace distribution,
#' \item \link[stats]{dlnorm} - Log normal distribution,
# \item dllaplace - Log Laplace distribution,
# \item dls - Log S distribution,
# \item dlgnorm - Log Generalised Normal distribution,
# \item \link[greybox]{dbcnorm} - Box-Cox normal distribution,
#' \item \link[statmod]{dinvgauss} - Inverse Gaussian distribution,
#' }
#'
#' For some more information about the model and its implementation, see the
#' vignette: \code{vignette("adam","smooth")}.
#'
#' The function \code{auto.adam()} tries out models with the specified
#' distributions and returns the one with the most suitable one.
#'
#' @template ssAuthor
#' @template ssKeywords
#'
#' @template ssADAMRef
#' @template ssGeneralRef
#' @template ssIntermittentRef
#' @template ssETSRef
#' @template ssIntervalsRef
#'
#' @param data Vector, containing data needed to be forecasted. If a matrix (or
#' data.frame / data.table) is provided, then the first column is used as a
#' response variable, while the rest of the matrix is used as a set of explanatory
#' variables. \code{formula} can be used in the latter case in order to define what
#' relation to have.
#' @param model The type of ETS model. The first letter stands for the type of
#' the error term ("A" or "M"), the second (and sometimes the third as well) is for
#' the trend ("N", "A", "Ad", "M" or "Md"), and the last one is for the type of
#' seasonality ("N", "A" or "M"). In case of several lags, the seasonal components
#' are assumed to be the same. The model is then printed out as
#' ETS(M,Ad,M)[m1,m2,...], where m1, m2, ... are the lags specified by the
#' \code{lags} parameter.
#' There are several options for the \code{model} besides the conventional ones,
#' which rely on information criteria:
#' \enumerate{
#' \item \code{model="ZZZ"} means that the model will be selected based on the
#' chosen information criteria type. The Branch and Bound is used in the process.
#' \item \code{model="XXX"} means that only additive components are tested, using
#' Branch and Bound.
#' \item \code{model="YYY"} implies selecting between multiplicative components.
#' \item \code{model="CCC"} triggers the combination of forecasts of models using
#' information criteria weights (Kolassa, 2011).
#' \item combinations between these four and the classical components are also
#' accepted. For example, \code{model="CAY"} will combine models with additive
#' trend and either none or multiplicative seasonality.
#' \item \code{model="PPP"} will produce the selection between pure additive and
#' pure multiplicative models. "P" stands for "Pure". This cannot be mixed with
#' other types of components.
#' \item \code{model="FFF"} will select between all the 30 types of models. "F"
#' stands for "Full". This cannot be mixed with other types of components.
#' \item The parameter \code{model} can also be a vector of names of models for a
#' finer tuning (pool of models). For example, \code{model=c("ANN","AAA")} will
#' estimate only two models and select the best of them.
#' }
#'
#' Also, \code{model} can accept a previously estimated adam and use all
#' its parameters.
#'
#' Keep in mind that model selection with "Z" components uses Branch and Bound
#' algorithm and may skip some models that could have slightly smaller
#' information criteria. If you want to do a exhaustive search, you would need
#' to list all the models to check as a vector.
#'
#' The default value is set to \code{"ZXZ"}, because the multiplicative trend is explosive
#' and dangerous. It should be used only for each separate time series, not for the
#' automated predictions for big  datasets.
#'
#' @param lags Defines lags for the corresponding components. All components
#' count, starting from level, so ETS(M,M,M) model for monthly data will have
#' \code{lags=c(1,1,12)}. However, the function will also accept \code{lags=c(12)},
#' assuming that the lags 1 were dropped.
#' @param orders The order of ARIMA to be included in the model. This should be passed
#' either as a vector (in which case the non-seasonal ARIMA is assumed) or as a list of
#' a type \code{orders=list(ar=c(p,P),i=c(d,D),ma=c(q,Q))}, in which case the \code{lags}
#' variable is used in order to determine the seasonality m. See \link[smooth]{msarima}
#' for details. Note that ARIMA here is mainly treated as an addition to the ETS model and
#' does not allow constant / drift.
#'
#' In case of \code{auto.adam()} function, \code{orders} accepts one more parameters:
#' \code{orders=list(select=FALSE)}. If \code{TRUE}, then the function will select the most
#' appropriate order using a mechanism similar to \code{auto.msarima()}. The values
#' \code{list(ar=...,i=...,ma=...)} specify the maximum orders to check in this case.
#' @param formula Formula to use in case of explanatory variables. If \code{NULL},
#' then all the variables are used as is. Only considered if \code{data} is a matrix and
#' \code{regressors="use"}.
#' @param distribution what density function to assume for the error term. The full
#' name of the distribution should be provided, starting with the letter "d" -
#' "density". The names align with the names of distribution functions in R.
#' For example, see \link[stats]{dnorm}. For detailed explanation of available
#' distributions, see vignette in greybox package: \code{vignette("greybox","alm")}.
#' @param loss The type of Loss Function used in optimization. \code{loss} can
#' be:
#' \itemize{
#' \item \code{likelihood} - the model is estimated via the maximisation of the
#' likelihood of the function specified in \code{distribution};
#' \item \code{MSE} (Mean Squared Error),
#' \item \code{MAE} (Mean Absolute Error),
#' \item \code{HAM} (Half Absolute Moment),
#' \item \code{LASSO} - use LASSO to shrink the parameters of the model;
#' \item \code{RIDGE} - use RIDGE to shrink the parameters of the model;
#' \item \code{TMSE} - Trace Mean Squared Error,
#' \item \code{GTMSE} - Geometric Trace Mean Squared Error,
#' \item \code{MSEh} - optimisation using only h-steps ahead error,
#' \item \code{MSCE} - Mean Squared Cumulative Error.
#' }
#' In case of LASSO / RIDGE, the variables are not normalised prior to the estimation,
#' but the parameters are divided by the mean values of explanatory variables.
#'
#' Note that model selection and combination works properly only for the default
#' \code{loss="likelihood"}.
#'
#' Furthermore, just for fun the absolute and half analogues of multistep estimators
#' are available: \code{MAEh}, \code{TMAE}, \code{GTMAE}, \code{MACE},
#' \code{HAMh}, \code{THAM}, \code{GTHAM}, \code{CHAM}.
#'
#' Last but not least, user can provide their own function here as well, making sure
#' that it accepts parameters \code{actual}, \code{fitted} and \code{B}. Here is an
#' example:
#' \code{lossFunction <- function(actual, fitted, B) return(mean(abs(actual-fitted)))}
#' \code{loss=lossFunction}
#' @param h The forecast horizon. Mainly needed for the multistep loss functions.
#' @param holdout Logical. If \code{TRUE}, then the holdout of the size \code{h}
#' is taken from the data (can be used for the model testing purposes).
#' @param persistence Persistence vector \eqn{g}, containing smoothing
#' parameters. If \code{NULL}, then estimated. Can be also passed as a names list of
#' the type: \code{persistence=list(level=0.1, trend=0.05, seasonal=c(0.1,0.2),
#' xreg=c(0.1,0.2))}. Dropping some elements from the named list will make the function
#' estimate them. e.g. if you don't specify seasonal in the persistence for the ETS(M,N,M)
#' model, it will be estimated.
#' @param phi Value of damping parameter. If \code{NULL} then it is estimated.
#' Only applicable for damped-trend models.
#' @param initial Can be either character or a list, or a vector of initial states.
#' If it is character, then it can be \code{"optimal"}, meaning that the initial
#' states are optimised, or \code{"backcasting"}, meaning that the initials are
#' produced using backcasting procedure (advised for data with high frequency). In
#' case of the list, it is recommended to use the named one and to provide those
#' initial components that are available. For example:
#' \code{initial=list(level=1000,trend=10,seasonal=list(c(1,2),c(1,2,3,4)),
#' arima=1,xreg=100)}. If some of the components are needed by the model, but are
#' not provided in the list, they will be estimated. If the vector is provided,
#' then it is expected that the components will be provided one after another
#' without any gaps.
#' @param arma Either the named list or a vector with AR / MA parameters ordered lag-wise.
#' The number of elements should correspond to the specified orders e.g.
#' \code{orders=list(ar=c(1,1),ma=c(1,1)), lags=c(1,4), arma=list(ar=c(0.9,0.8),ma=c(-0.3,0.3))}
#' @param occurrence The type of model used in probability estimation. Can be
#' \code{"none"} - none,
#' \code{"fixed"} - constant probability,
#' \code{"general"} - the general Beta model with two parameters,
#' \code{"odds-ratio"} - the Odds-ratio model with b=1 in Beta distribution,
#' \code{"inverse-odds-ratio"} - the model with a=1 in Beta distribution,
#' \code{"direct"} - the TSB-like (Teunter et al., 2011) probability update
#' mechanism a+b=1,
#' \code{"auto"} - the automatically selected type of occurrence model.
#'
#' The type of model used in the occurrence is equal to the one provided in the
#' \code{model} parameter.
#'
#' Also, a model produced using \link[smooth]{oes} or \link[greybox]{alm} function
#' can be used here.
#' @param ic The information criterion to use in the model selection / combination
#' procedure.
#' @param bounds The type of bounds for the persistence to use in the model
#' estimation. Can be either \code{admissible} - guaranteeing the stability of the
#' model, \code{traditional} - restricting the values with (0, 1) or \code{none} - no
#' restrictions (potentially dangerous).
#' @param regressors The variable defines what to do with the provided xreg:
#' \code{"use"} means that all of the data should be used, while
#' \code{"select"} means that a selection using \code{ic} should be done,
#' \code{"adapt"} will trigger the mechanism of time varying parameters for the
#' explanatory variables.
#' @param silent Specifies, whether to provide the progress of the function or not.
#' If \code{TRUE}, then the function will print what it does and how much it has
#' already done.
#' @param ...  Other non-documented parameters. For example, \code{FI=TRUE} will
#' make the function also produce Fisher Information matrix, which then can be
#' used to calculated variances of smoothing parameters and initial states of
#' the model. This is calculated based on the hessian of log-likelihood function and
#' accepts \code{stepSize} parameter, determining how it is calculated. The default value
#' is \code{stepSize=.Machine$double.eps^(1/4)}. This is used in the \link[stats]{vcov} method.
#' Starting values of parameters can be passed via \code{B}, while the upper and lower
#' bounds should be passed in \code{ub} and \code{lb} respectively. In this case they
#' will be used for optimisation. These values should have the length equal
#' to the number of parameters to estimate in the following order:
#' \enumerate{
#' \item All smoothing parameters (for the states and then for the explanatory variables);
#' \item Damping parameter (if needed);
#' \item ARMA parameters;
#' \item All the initial values (for the states and then for the explanatory variables).
#' }
#' You can also pass parameters to the optimiser in order to fine tune its work:
#' \itemize{
#' \item \code{maxeval} - maximum number of evaluations to carry out. The default is 40 per
#' estimated parameter, at least 1000 if pure ARIMA is considered and at least 500 if
#' explanatory variables are introduced in the model;
#' \item \code{maxtime} - stop, when the optimisation time (in seconds) exceeds this;
#' \item \code{xtol_rel} - the relative precision of the optimiser (the default is 1E-6);
#' \item \code{xtol_abs} - the absolute precision of the optimiser (the default is 1E-8);
#' \item \code{ftol_rel} - the stopping criterion in case of the relative change in the loss
#' function (the default is 1E-8);
#' \item \code{ftol_abs} - the stopping criterion in case of the absolute change in the loss
#' function (the default is 0 - not used);
#' \item \code{algorithm} - the algorithm to use in optimisation
#' (by default, \code{"NLOPT_LN_SBPLX"} is used);
#' \item \code{print_level} - the level of output for the optimiser (0 by default).
#' If equal to 41, then the detailed results of the optimisation are returned.
#' }
#' You can read more about these parameters by running the function
#' \link[nloptr]{nloptr.print.options}.
#' Finally, the parameter \code{lambda} for LASSO / RIDGE, \code{alpha} for the Asymmetric
#' Laplace, \code{beta} for the Generalised Normal and \code{nu} for Student's distributions
#' can be provided here as well.
#'
#' @return Object of class "adam" is returned. It contains the list of the
#' following values:
#' \itemize{
#' \item \code{model} - the name of the constructed model,
#' \item \code{timeElapsed} - the time elapsed for the estimation of the model,
#' \item \code{data} - the in-sample part of the data used for the training of the model. Includes
#' the actual values in the first column,
#' \item \code{holdout} - the holdout part of the data, excluded for purposes of model evaluation,
#' \item \code{fitted} - the vector of fitted values,
#' \item \code{residuals} - the vector of residuals,
#' \item \code{forecast} - the point forecast for h steps ahead (by default NA is returned),
#' \item \code{states} - the matrix of states with observations in rows and states in columns,
#' \item \code{persisten} - the vector of smoothing parameters,
#' \item \code{phi} - the value of damping parameter,
#' \item \code{transition} - the transition matrix,
#' \item \code{measurement} - the measurement matrix with observations in rows and state elements
#' in columns,
#' \item \code{initial} - the named list of initial values, including level, trend, seasonal, ARIMA
#' and xreg components,
#' \item \code{initialEstimated} - the named vector, defining which of the initials were estimated in
#' the model,
#' \item \code{initialType} - the type of initialisation used ("optimal" / "backcasting" / "provided"),
#' \item \code{orders} - the orders of ARIMA used in the estimation,
#' \item \code{arma} - the list of AR / MA parameters used in the model,
#' \item \code{nParam} - the matrix of the estimated / provided parameters,
#' \item \code{occurrence} - the oes model used for the occurrence part of the model,
#' \item \code{formula} - the formula used for the explanatory variables expansion,
#' \item \code{loss} - the type of loss function used in the estimation,
#' \item \code{lossValue} - the value of that loss function,
#' \item \code{logLik} - the value of the log-likelihood,
#' \item \code{distribution} - the distribution function used in the calculation of the likelihood,
#' \item \code{scale} - the value of the scale parameter,
#' \item \code{lambda} - the value of the parameter used in LASSO / dalaplace / dt,
#' \item \code{B} - the vector of all estimated parameters,
#' \item \code{lags} - the vector of lags used in the model construction,
#' \item \code{lagsAll} - the vector of the internal lags used in the model,
#' \item \code{profile} - the matrix with the profile used in the construction of the model,
#' \item \code{call} - the call used in the evaluation,
#' \item \code{bounds} - the type of bounds used in the process,
#' \item \code{other} - the list with other parameters, such as shape for distributions or ARIMA
#' polynomials.
#' }
#'
#' @seealso \code{\link[forecast]{ets}, \link[smooth]{es}}
#'
#' @examples
#'
#' ### The main examples are provided in the adam vignette, check it out via:
#' # vignette("adam","smooth")
#'
#' # Model selection using a specified pool of models
#' ourModel <- adam(rnorm(100,100,10), model=c("ANN","ANA","AAA"), lags=c(5,10))
#'
#' summary(ourModel)
#' forecast(ourModel)
#' par(mfcol=c(3,4))
#' plot(ourModel, c(1:11))
#'
#' # Model combination using a specified pool
#' ourModel <- adam(rnorm(100,100,10), model=c("ANN","AAN","MNN","CCC"), lags=c(5,10))
#'
#' # ADAM ARIMA
#' ourModel <- adam(rnorm(100,100,10), model="NNN",
#'                  lags=c(1,4), orders=list(ar=c(1,0),i=c(1,0),ma=c(1,1)))
#'
#' @importFrom forecast forecast na.interp
#' @importFrom greybox dlaplace dalaplace ds stepwise alm is.occurrence is.alm polyprod
#' @importFrom stats dnorm dlogis dt dlnorm frequency confint vcov predict
#' @importFrom stats formula update model.frame model.matrix contrasts setNames terms
#' @importFrom statmod dinvgauss
#' @importFrom nloptr nloptr
#' @importFrom pracma hessian
#' @importFrom zoo zoo
#' @importFrom utils head
#' @rdname adam
#' @export adam
adam <- function(data, model="ZXZ", lags=c(1,frequency(data)), orders=list(ar=c(0),i=c(0),ma=c(0)),
                 formula=NULL, regressors=c("use","select","adapt"),
                 distribution=c("default","dnorm","dlaplace","ds","dgnorm","dalaplace",
                                "dlnorm","dinvgauss"),
                 loss=c("likelihood","MSE","MAE","HAM","LASSO","RIDGE","MSEh","TMSE","GTMSE","MSCE"),
                 h=0, holdout=FALSE,
                 persistence=NULL, phi=NULL, initial=c("optimal","backcasting"), arma=NULL,
                 occurrence=c("none","auto","fixed","general","odds-ratio","inverse-odds-ratio","direct"),
                 ic=c("AICc","AIC","BIC","BICc"), bounds=c("usual","admissible","none"),
                 silent=TRUE, ...){
    # Copyright (C) 2019 - Inf  Ivan Svetunkov

    # Start measuring the time of calculations
    startTime <- Sys.time();

    cl <- match.call();
    ellipsis <- list(...);
    # Assume that the model is not provided
    profilesRecentProvided <- FALSE;
    profilesRecentTable <- NULL;

    # paste0() is needed in order to get rid of potential issues with names
    yName <- paste0(deparse(substitute(data)),collapse="");

    # If a previous model is provided as a model, write down the variables
    if(is.adam(model) || is.adam.sim(model)){
        # If this is the simulated data, extract the parameters
        # if(is.adam.sim(model) & !is.null(dim(model$data))){
        #     warning("The provided model has several submodels. Choosing a random one.",call.=FALSE);
        #     i <- round(runif(1,1:length(model$persistence)));
        #     persistence <- model$persistence[,i];
        #     initial <- model$initial[,i];
        #     initialSeason <- model$initialSeason[,i];
        #     if(any(model$iprob!=1)){
        #         occurrence <- "a";
        #     }
        # }
        # else{
        persistence <- model$persistence;
        initial <- model$initial;
        initialEstimated <- model$initialEstimated;
        distribution <- model$distribution;
        loss <- model$loss;
        persistence <- model$persistence;
        phi <- model$phi;
        if(model$initialType!="backcasting"){
            initial <- model$initial;
        }
        else{
            initial <- "b";
        }
        occurrence <- model$occurrence;
        ic <- model$ic;
        bounds <- model$bounds;
        # lambda for LASSO
        ellipsis$lambda <- model$other$lambda;
        # parameters for distributions
        ellipsis$alpha <- model$other$alpha;
        ellipsis$beta <- model$other$beta;
        ellipsis$nu <- model$other$nu;
        ellipsis$B <- model$B;
        CFValue <- model$lossValue;
        logLikADAMValue <- logLik(model);
        profilesRecentTable <- t(model$states);
        profilesRecentProvided[] <- TRUE;
        regressors <- model$regressors;
        if(is.null(formula)){
            formula <- formula(model);
        }

        # Parameters of the original ARIMA model
        lags <- lags(model);
        orders <- orders(model);
        arma <- model$arma;

        model <- modelType(model);
        modelDo <- "use";
        # if(any(unlist(gregexpr("C",model))!=-1)){
        #     initial <- "o";
        # }
    }
    else if(inherits(model,"ets")){
        # Extract smoothing parameters
        i <- 1;
        lags <- 1;
        persistence <- coef(model)[i];
        if(model$components[2]!="N"){
            i <- i+1;
            persistence <- c(persistence,coef(model)[i]);
            if(model$components[3]!="N"){
                i <- i+1;
                persistence <- c(persistence,coef(model)[i]);
            }
        }
        else{
            if(model$components[3]!="N"){
                i <- i+1;
                persistence <- c(persistence,coef(model)[i]);
            }
        }

        # Damping parameter
        if(model$components[4]=="TRUE"){
            i <- i+1;
            phi <- coef(model)[i];
        }

        # Initials
        i <- i+1;
        initial <- coef(model)[i];
        # Initial for the trend
        if(model$components[2]!="N"){
            i <- i+1;
            lags <- c(lags,1);
            initial <- c(initial,coef(model)[i]);
        }

        # Initials of seasonal component
        if(model$components[3]!="N"){
            if(model$components[2]!="N"){
                initial <- c(initial,rev(model$states[1,-c(1:2)]));
            }
            else{
                initial <- c(initial,rev(model$states[1,-c(1)]));
            }
            lags <- c(lags,model$m);
        }
        model <- modelType(model);
        distribution <- "dnorm";
        loss <- "likelihood";
        modelDo <- "use"
    }
    else if(is.character(model)){
        modelDo <- "";
        # Everything is okay
    }
    else{
        modelDo <- "";
        warning("A model of an unknown class was provided. Switching to 'ZZZ'.",call.=FALSE);
        model <- "ZZZ";
    }

    #### Check the parameters of the function and create variables based on them ####
    checkerReturn <- parametersChecker(data, model, lags, formula, orders, arma,
                                       persistence, phi, initial,
                                       distribution, loss, h, holdout, occurrence, ic, bounds,
                                       regressors, yName,
                                       silent, modelDo, ParentEnvironment=environment(), ellipsis, fast=FALSE);

    #### Return regression if it is pure ####
    if(is.alm(checkerReturn)){
        obsInSample <- nobs(checkerReturn);
        nParam <- length(checkerReturn$coefficient);

        modelReturned <- list(model="Regression");
        modelReturned$timeElapsed <- Sys.time()-startTime;
        if(is.null(formula)){
            formula <- formula(checkerReturn);
        }
        if(holdout){
            # Robustify the names of variables
            colnames(data) <- make.names(colnames(data),unique=TRUE);
            modelReturned$holdout <- data[obsInSample+c(1:h),,drop=FALSE];
        }
        else{
            modelReturned$holdout <- NULL;
        }
        responseName <- all.vars(formula)[1];
        y <- data[,responseName];
        # Extract indeces from the data
        yIndex <- try(time(y),silent=TRUE);
        # If we cannot extract time, do something
        if(inherits(yIndex,"try-error")){
            if(!is.data.frame(data) && !is.null(dim(data))){
                yIndex <- as.POSIXct(rownames(data));
            }
            else if(is.data.frame(data)){
                yIndex <- c(1:nrow(data));
            }
            else{
                yIndex <- c(1:length(data));
            }
        }
        # Prepare fitted, residuals and the forecasts
        if(inherits(y ,"zoo")){
            modelReturned$data <- data[1:obsInSample,,drop=FALSE];
            modelReturned$fitted <- zoo(fitted(checkerReturn), order.by=yIndex[1:obsInSample]);
            modelReturned$residuals <- zoo(residuals(checkerReturn), order.by=yIndex[1:obsInSample]);
            # If we need to forecast and we had holdout=FALSE...
            if(h>0){
                if(holdout){
                    modelReturned$forecast <- zoo(forecast(checkerReturn,h=h,newdata=tail(data,h))$mean,
                                                  order.by=yIndex[obsInSample+1:h]);
                }
                else{
                    modelReturned$forecast <- zoo(forecast(checkerReturn,h=h)$mean,
                                                  order.by=yIndex[obsInSample+1:h]);
                }
            }
            else{
                modelReturned$forecast <- zoo(NA, order.by=yIndex[obsInSample+1]);
            }
            modelReturned$states <- zoo(matrix(coef(checkerReturn), obsInSample+1, nParam, byrow=TRUE,
                                               dimnames=list(NULL, names(coef(checkerReturn)))),
                                        order.by=c(yIndex[1]-diff(yIndex[1:2]),yIndex[1:obsInSample]));
        }
        else{
            yFrequency <- frequency(y);
            modelReturned$data <- ts(data[1:obsInSample,,drop=FALSE], start=yIndex[1], frequency=yFrequency);
            modelReturned$fitted <- ts(fitted(checkerReturn), start=yIndex[1], frequency=yFrequency);
            modelReturned$residuals <- ts(residuals(checkerReturn), start=yIndex[1], frequency=yFrequency);
            if(h>0){
                if(holdout){
                    modelReturned$forecast <- ts(forecast(checkerReturn,h=h,newdata=tail(data,h))$mean,
                                                 start=yIndex[obsInSample+1], frequency=yFrequency);
                }
                else{
                    modelReturned$forecast <- zoo(forecast(checkerReturn,h=h)$mean,
                                                  order.by=yIndex[obsInSample]+diff(yIndex[1:2])*c(1:h));
                }
            }
            else{
                modelReturned$forecast <- ts(NA, start=yIndex[obsInSample]+diff(yIndex[1:2]), frequency=yFrequency);
            }
            modelReturned$states <- ts(matrix(coef(checkerReturn), obsInSample+1, nParam, byrow=TRUE,
                                           dimnames=list(NULL, names(coef(checkerReturn)))),
                                       start=yIndex-diff(yIndex[1:2]), frequency=yFrequency);
        }
        modelReturned$persistence <- rep(0, nParam);
        names(modelReturned$persistence) <- paste0("delta",c(1:nParam));
        modelReturned$phi <- 1;
        modelReturned$transition <- diag(nParam);
        modelReturned$measurement <- checkerReturn$data;
        modelReturned$measurement[,1] <- 1;
        colnames(modelReturned$measurement) <- colnames(modelReturned$states);
        modelReturned$initial <- list(xreg=coef(checkerReturn));
        modelReturned$initialType <- "optimal";
        modelReturned$initialEstimated <- TRUE;
        names(modelReturned$initialEstimated) <- "xreg";
        modelReturned$orders <- list(ar=0,i=0,ma=0);
        modelReturned$arma <- NULL;
        # Number of estimated parameters
        parametersNumber <- matrix(0,2,4,
                                   dimnames=list(c("Estimated","Provided"),
                                                 c("nParamInternal","nParamXreg","nParamOccurrence","nParamAll")));
        parametersNumber[1,2] <- nParam;
        if(is.occurrence(checkerReturn$occurrence)){
            parametersNumber[1,3] <- nParam;
        }
        parametersNumber[1,4] <- sum(parametersNumber[1,1:3]);
        modelReturned$nParam <- parametersNumber;
        modelReturned$occurrence <- checkerReturn$occurrence;
        modelReturned$formula <- formula(checkerReturn);
        modelReturned$regressors <- "use";
        modelReturned$loss <- checkerReturn$loss;
        modelReturned$lossValue <- checkerReturn$lossValue;
        modelReturned$lossFunction <- checkerReturn$lossFunction;
        modelReturned$logLik <- logLik(checkerReturn);
        modelReturned$distribution <- checkerReturn$distribution;
        modelReturned$scale <- checkerReturn$scale;
        modelReturned$other <- checkerReturn$other;
        modelReturned$B <- coef(checkerReturn);
        modelReturned$lags <- 1;
        modelReturned$lagsAll <- rep(1,nParam);
        modelReturned$FI <- checkerReturn$FI;
        if(holdout){
            # This won't work if transformations of the response variable are done...
            modelReturned$accuracy <- measures(modelReturned$holdout[,responseName],modelReturned$forecast,
                                               modelReturned$data[,responseName]);
        }
        else{
            modelReturned$accuracy <- NULL;
        }
        class(modelReturned) <- c("adam","smooth");
        if(!silent){
            plot(modelReturned,7)
        }

        return(modelReturned);
    }

    #### The function creates the technical variables (lags etc) based on the type of the model ####
    architector <- function(etsModel, Etype, Ttype, Stype, lags, lagsModelSeasonal,
                            xregNumber, obsInSample, initialType,
                            arimaModel, lagsModelARIMA, xregModel,
                            profilesRecentTable=NULL, profilesRecentProvided=FALSE){
        # If there is ETS
        if(etsModel){
            modelIsTrendy <- Ttype!="N";
            if(modelIsTrendy){
                # Make lags (1, 1)
                lagsModel <- matrix(c(1,1),ncol=1);
                componentsNamesETS <- c("level","trend");
            }
            else{
                # Make lags (1, ...)
                lagsModel <- matrix(c(1),ncol=1);
                componentsNamesETS <- c("level");
            }
            modelIsSeasonal <- Stype!="N";
            if(modelIsSeasonal){
                # If the lags are for the non-seasonal model
                lagsModel <- matrix(c(lagsModel,lagsModelSeasonal),ncol=1);
                componentsNumberETSSeasonal <- length(lagsModelSeasonal);
                if(componentsNumberETSSeasonal>1){
                    componentsNamesETS <- c(componentsNamesETS,paste0("seasonal",c(1:componentsNumberETSSeasonal)));
                }
                else{
                    componentsNamesETS <- c(componentsNamesETS,"seasonal");
                }
            }
            else{
                componentsNumberETSSeasonal <- 0;
            }
            lagsModelAll <- lagsModel;

            componentsNumberETS <- length(lagsModel);
        }
        else{
            modelIsTrendy <- modelIsSeasonal <- FALSE;
            componentsNumberETS <- componentsNumberETSSeasonal <- 0;
            componentsNamesETS <- NULL;
            lagsModelAll <- lagsModel <- NULL;
        }

        # If there is ARIMA
        if(arimaModel){
            lagsModelAll <- matrix(c(lagsModel,lagsModelARIMA), ncol=1);
        }

        # If there are xreg
        if(xregModel){
            lagsModelAll <- matrix(c(lagsModelAll,rep(1,xregNumber)), ncol=1);
        }
        lagsModelMax <- max(lagsModelAll);

        # Define the number of cols that should be in the matvt
        obsStates <- obsInSample + lagsModelMax;

        # Create ADAM profiles for correct treatment of seasonality
        adamProfiles <- adamProfileCreator(lagsModelAll, lagsModelMax, obsAll,
                                           lags=lags, yIndex=yIndexAll, yClasses=yClasses);
        if(profilesRecentProvided){
            profilesRecentTable <- profilesRecentTable[,1:lagsModelMax,drop=FALSE];
        }
        else{
            profilesRecentTable <- adamProfiles$recent;
        }
        profilesObservedTable <- adamProfiles$observed;

        return(list(lagsModel=lagsModel,lagsModelAll=lagsModelAll, lagsModelMax=lagsModelMax,
                    componentsNumberETS=componentsNumberETS, componentsNumberETSSeasonal=componentsNumberETSSeasonal,
                    componentsNumberETSNonSeasonal=componentsNumberETS-componentsNumberETSSeasonal,
                    componentsNamesETS=componentsNamesETS, obsStates=obsStates, modelIsTrendy=modelIsTrendy,
                    modelIsSeasonal=modelIsSeasonal,
                    profilesObservedTable=profilesObservedTable, profilesRecentTable=profilesRecentTable));
    }

    #### The function creates the necessary matrices based on the model and provided parameters ####
    # This is needed in order to initialise the estimation
    creator <- function(etsModel, Etype, Ttype, Stype, modelIsTrendy, modelIsSeasonal,
                        lags, lagsModel, lagsModelARIMA, lagsModelAll, lagsModelMax,
                        profilesRecentTable=NULL, profilesRecentProvided=FALSE,
                        obsStates, obsInSample, obsAll, componentsNumberETS, componentsNumberETSSeasonal,
                        componentsNamesETS, otLogical, yInSample,
                        # Persistence
                        persistence, persistenceEstimate,
                        persistenceLevel, persistenceLevelEstimate, persistenceTrend, persistenceTrendEstimate,
                        persistenceSeasonal, persistenceSeasonalEstimate,
                        persistenceXreg, persistenceXregEstimate, persistenceXregProvided,
                        phi,
                        # Initials
                        initialType, initialEstimate,
                        initialLevel, initialLevelEstimate, initialTrend, initialTrendEstimate,
                        initialSeasonal, initialSeasonalEstimate,
                        initialArima, initialArimaEstimate, initialArimaNumber,
                        initialXregEstimate, initialXregProvided,
                        # ARIMA elements
                        arimaModel, arRequired, iRequired, maRequired, armaParameters,
                        arOrders, iOrders, maOrders,
                        componentsNumberARIMA, componentsNamesARIMA,
                        # Explanatory variables
                        xregModel, xregModelInitials, xregData, xregNumber, xregNames,
                        xregParametersPersistence){

        # Matrix of states. Time in columns, components in rows
        matVt <- matrix(NA, componentsNumberETS+componentsNumberARIMA+xregNumber, obsStates,
                        dimnames=list(c(componentsNamesETS,componentsNamesARIMA,xregNames),NULL));

        # Measurement rowvector
        matWt <- matrix(1, obsAll, componentsNumberETS+componentsNumberARIMA+xregNumber,
                        dimnames=list(NULL,c(componentsNamesETS,componentsNamesARIMA,xregNames)));

        # If xreg are provided, then fill in the respective values in Wt vector
        if(xregModel){
            matWt[,componentsNumberETS+componentsNumberARIMA+1:xregNumber] <- xregData;
        }

        # Transition matrix
        matF <- diag(componentsNumberETS+componentsNumberARIMA+xregNumber);

        # Persistence vector
        vecG <- matrix(0, componentsNumberETS+componentsNumberARIMA+xregNumber, 1,
                       dimnames=list(c(componentsNamesETS,componentsNamesARIMA,xregNames),NULL));

        j <- 0;
        # ETS model, persistence
        if(etsModel){
            j <- j+1;
            rownames(vecG)[j] <- "alpha";
            if(!persistenceLevelEstimate){
                vecG[j,] <- persistenceLevel;
            }
            if(modelIsTrendy){
                j <- j+1;
                rownames(vecG)[j] <- "beta";
                if(!persistenceTrendEstimate){
                    vecG[j,] <- persistenceTrend;
                }
            }
            if(modelIsSeasonal){
                if(!all(persistenceSeasonalEstimate)){
                    vecG[j+which(!persistenceSeasonalEstimate),] <- persistenceSeasonal;
                }
                if(componentsNumberETSSeasonal>1){
                    rownames(vecG)[j+c(1:componentsNumberETSSeasonal)] <- paste0("gamma",c(1:componentsNumberETSSeasonal));
                }
                else{
                    rownames(vecG)[j+1] <- "gamma";
                }
                j <- j+componentsNumberETSSeasonal;
            }
        }

        # ARIMA model, names for persistence
        if(arimaModel){
            # Remove diagonal from the ARIMA part of the matrix
            matF[j+1:componentsNumberARIMA,j+1:componentsNumberARIMA] <- 0;
            if(componentsNumberARIMA>1){
                rownames(vecG)[j+1:componentsNumberARIMA] <- paste0("psi",c(1:componentsNumberARIMA));
            }
            else{
                rownames(vecG)[j+1:componentsNumberARIMA] <- "psi";
            }
            j <- j+componentsNumberARIMA;
        }

        # Regression, persistence
        if(xregModel){
            if(persistenceXregProvided && !persistenceXregEstimate){
                vecG[j+1:xregNumber,] <- persistenceXreg;
            }
            rownames(vecG)[j+1:xregNumber] <- paste0("delta",xregParametersPersistence);
        }

        # Damping parameter value
        if(etsModel && modelIsTrendy){
            matF[1,2] <- phi;
            matF[2,2] <- phi;

            matWt[,2] <- phi;
        }

        # If the arma parameters were provided, fill in the persistence
        if(arimaModel && (!arEstimate && !maEstimate)){
            # Call polynomial
            arimaPolynomials <- polynomialiser(NULL, arOrders, iOrders, maOrders,
                                               arRequired, maRequired, arEstimate, maEstimate, armaParameters, lags);
            # Fill in the transition matrix
            if(nrow(nonZeroARI)>0){
                matF[componentsNumberETS+nonZeroARI[,2],componentsNumberETS+nonZeroARI[,2]] <-
                    -arimaPolynomials$ariPolynomial[nonZeroARI[,1]];
            }
            # Fill in the persistence vector
            if(nrow(nonZeroARI)>0){
                vecG[componentsNumberETS+nonZeroARI[,2]] <- -arimaPolynomials$ariPolynomial[nonZeroARI[,1]];
            }
            if(nrow(nonZeroMA)>0){
                vecG[componentsNumberETS+nonZeroMA[,2]] <- vecG[componentsNumberETS+nonZeroMA[,2]] +
                    arimaPolynomials$maPolynomial[nonZeroMA[,1]];
            }
        }
        else{
            arimaPolynomials <- NULL;
        }

        if(!profilesRecentProvided){
            # ETS model, initial state
            # If something needs to be estimated...
            if(etsModel){
                if(initialEstimate){
                    # For the seasonal models
                    if(modelIsSeasonal){
                        if(obsNonzero>=lagsModelMax*2){
                            # If either Etype or Stype are multiplicative, do multiplicative decomposition
                            decompositionType <- c("additive","multiplicative")[any(c(Etype,Stype)=="M")+1];
                            yDecomposition <- msdecompose(yInSample, lags[lags!=1], type=decompositionType);
                            j <- 1;
                            # level
                            if(initialLevelEstimate){
                                matVt[j,1:lagsModelMax] <- yDecomposition$initial[1];
                                # matVt[j,1:lagsModelMax] <- mean(yInSample[1:lagsModelMax]);
                                if(xregModel){
                                    if(Etype=="A"){
                                        matVt[j,1:lagsModelMax] <- matVt[j,1:lagsModelMax] -
                                            as.vector(xregModelInitials[[1]]$initialXreg %*% xregData[1,]);
                                    }
                                    else{
                                        matVt[j,1:lagsModelMax] <- matVt[j,1:lagsModelMax] /
                                            as.vector(exp(xregModelInitials[[2]]$initialXreg %*% xregData[1,]));
                                    }
                                }
                            }
                            else{
                                matVt[j,1:lagsModelMax] <- initialLevel;
                            }
                            j <- j+1;
                            # If trend is needed
                            if(modelIsTrendy){
                                if(initialTrendEstimate){
                                    if(Ttype=="A" && Stype=="M"){
                                        # if(initialLevelEstimate){
                                        #     # level fix
                                        #     matVt[j-1,1:lagsModelMax] <- exp(mean(log(yInSample[otLogical][1:lagsModelMax])));
                                        # }
                                        # trend
                                        matVt[j,1:lagsModelMax] <- prod(yDecomposition$initial)-yDecomposition$initial[1];
                                        # If the initial trend is higher than the lowest value, initialise with zero.
                                        # This is a failsafe mechanism for the mixed models
                                        if(matVt[j,1]<0 && abs(matVt[j,1])>min(abs(yInSample[otLogical]))){
                                            matVt[j,1:lagsModelMax] <- 0;
                                        }
                                    }
                                    else if(Ttype=="M" && Stype=="A"){
                                        # if(initialLevelEstimate){
                                        #     # level fix
                                        #     matVt[j-1,1:lagsModelMax] <- exp(mean(log(yInSample[otLogical][1:lagsModelMax])));
                                        # }
                                        # trend
                                        matVt[j,1:lagsModelMax] <- sum(abs(yDecomposition$initial))/abs(yDecomposition$initial[1]);
                                    }
                                    else if(Ttype=="M"){
                                        # trend is too dangerous, make it start from 1.
                                        matVt[j,1:lagsModelMax] <- 1;
                                    }
                                    else{
                                        # trend
                                        matVt[j,1:lagsModelMax] <- yDecomposition$initial[2];
                                    }
                                    # This is a failsafe for multiplicative trend models, so that the thing does not explode
                                    if(Ttype=="M" && any(matVt[j,1:lagsModelMax]>1.1)){
                                        matVt[j,1:lagsModelMax] <- 1;
                                    }
                                    # This is a failsafe for multiplicative trend models, so that the thing does not explode
                                    if(Ttype=="M" && any(matVt[1,1:lagsModelMax]<0)){
                                        matVt[1,1:lagsModelMax] <- yInSample[otLogical][1];
                                    }
                                }
                                else{
                                    matVt[j,1:lagsModelMax] <- initialTrend;
                                }
                                j <- j+1;
                            }
                            #### Seasonal components
                            # For pure models use stuff as is
                            if(all(c(Etype,Stype)=="A") || all(c(Etype,Stype)=="M") ||
                               (Etype=="A" & Stype=="M")){
                                for(i in 1:componentsNumberETSSeasonal){
                                    if(initialSeasonalEstimate[i]){
                                        matVt[i+j-1,1:lagsModel[i+j-1]] <- yDecomposition$seasonal[[i]];
                                        # Renormalise the initial seasons
                                        if(Stype=="A"){
                                            matVt[i+j-1,1:lagsModel[i+j-1]] <-
                                                matVt[i+j-1,1:lagsModel[i+j-1]] -
                                                mean(matVt[i+j-1,1:lagsModel[i+j-1]]);
                                        }
                                        else{
                                            matVt[i+j-1,1:lagsModel[i+j-1]] <-
                                                matVt[i+j-1,1:lagsModel[i+j-1]] /
                                                exp(mean(log(matVt[i+j-1,1:lagsModel[i+j-1]])));
                                        }
                                    }
                                    else{
                                        matVt[i+j-1,1:lagsModel[i+j-1]] <- initialSeasonal[[i]];
                                    }
                                }
                            }
                            # For mixed models use a different set of initials
                            else if(Etype=="M" && Stype=="A"){
                                for(i in 1:componentsNumberETSSeasonal){
                                    if(initialSeasonalEstimate[i]){
                                        matVt[i+j-1,1:lagsModel[i+j-1]] <- log(yDecomposition$seasonal[[i]])*min(yInSample[otLogical]);
                                        # Renormalise the initial seasons
                                        if(Stype=="A"){
                                            matVt[i+j-1,1:lagsModel[i+j-1]] <- matVt[i+j-1,1:lagsModel[i+j-1]] -
                                                mean(matVt[i+j-1,1:lagsModel[i+j-1]]);
                                        }
                                        else{
                                            matVt[i+j-1,1:lagsModel[i+j-1]] <- matVt[i+j-1,1:lagsModel[i+j-1]] /
                                                exp(mean(log(matVt[i+j-1,1:lagsModel[i+j-1]])));
                                        }
                                    }
                                    else{
                                        matVt[i+j-1,1:lagsModel[i+j-1]] <- initialSeasonal[[i]];
                                    }
                                }
                            }
                        }
                        else{
                            # If either Etype or Stype are multiplicative, do multiplicative decomposition
                            j <- 1;
                            # level
                            if(initialLevelEstimate){
                                matVt[j,1:lagsModelMax] <- mean(yInSample[1:lagsModelMax]);
                                if(xregModel){
                                    if(Etype=="A"){
                                        matVt[j,1:lagsModelMax] <- matVt[j,1:lagsModelMax] -
                                            as.vector(xregModelInitials[[1]]$initialXreg %*% xregData[1,]);
                                    }
                                    else{
                                        matVt[j,1:lagsModelMax] <- matVt[j,1:lagsModelMax] /
                                            as.vector(exp(xregModelInitials[[2]]$initialXreg %*% xregData[1,]));
                                    }
                                }
                            }
                            else{
                                matVt[j,1:lagsModelMax] <- initialLevel;
                            }
                            j <- j+1;
                            if(modelIsTrendy){
                                if(initialTrendEstimate){
                                    if(Ttype=="A"){
                                        # trend
                                        matVt[j,1:lagsModelMax] <- yInSample[2]-yInSample[1];
                                    }
                                    else if(Ttype=="M"){
                                        if(initialLevelEstimate){
                                            # level fix
                                            matVt[j-1,1:lagsModelMax] <- exp(mean(log(yInSample[otLogical][1:lagsModelMax])));
                                        }
                                        # trend
                                        matVt[j,1:lagsModelMax] <- yInSample[2]/yInSample[1];
                                    }
                                    # This is a failsafe for multiplicative trend models, so that the thing does not explode
                                    if(Ttype=="M" && matVt[j,1:lagsModelMax]>1.1){
                                        matVt[j,1:lagsModelMax] <- 1;
                                    }
                                }
                                else{
                                    matVt[j,1:lagsModelMax] <- initialTrend;
                                }

                                # Do roll back. Especially useful for backcasting and multisteps
                                if(Ttype=="A"){
                                    matVt[j-1,1:lagsModelMax] <- matVt[j-1,1] - matVt[j,1]*lagsModelMax;
                                }
                                else if(Ttype=="M"){
                                    matVt[j-1,1:lagsModelMax] <- matVt[j-1,1] / matVt[j,1]^lagsModelMax;
                                }
                                j <- j+1;
                            }
                            #### Seasonal components
                            # For pure models use stuff as is
                            if(Stype=="A"){
                                for(i in 1:componentsNumberETSSeasonal){
                                    if(initialSeasonalEstimate[i]){
                                        matVt[i+j-1,1:lagsModel[i+j-1]] <- yInSample[1:lagsModel[i+j-1]]-matVt[1,1];
                                        # Renormalise the initial seasons
                                        matVt[i+j-1,1:lagsModel[i+j-1]] <- matVt[i+j-1,1:lagsModel[i+j-1]] -
                                            mean(matVt[i+j-1,1:lagsModel[i+j-1]]);
                                    }
                                    else{
                                        matVt[i+j-1,1:lagsModel[i+j-1]] <- initialSeasonal[[i]];
                                    }
                                }
                            }
                            # For mixed models use a different set of initials
                            else{
                                for(i in 1:componentsNumberETSSeasonal){
                                    if(initialSeasonalEstimate[i]){
                                        matVt[i+j-1,1:lagsModel[i+j-1]] <- yInSample[1:lagsModel[i+j-1]]/matVt[1,1];
                                        # Renormalise the initial seasons
                                        matVt[i+j-1,1:lagsModel[i+j-1]] <- matVt[i+j-1,1:lagsModel[i+j-1]] /
                                            exp(mean(log(matVt[i+j-1,1:lagsModel[i+j-1]])));
                                    }
                                    else{
                                        matVt[i+j-1,1:lagsModel[i+j-1]] <- initialSeasonal[[i]];
                                    }
                                }
                            }
                        }
                    }
                    # Non-seasonal models
                    else{
                        # level
                        if(initialLevelEstimate){
                            matVt[1,1:lagsModelMax] <- mean(yInSample[1:max(lagsModelMax,ceiling(obsInSample*0.2))]);
                            if(xregModel){
                                if(Etype=="A"){
                                    matVt[1,1:lagsModelMax] <- matVt[1,lagsModelMax] -
                                        as.vector(xregModelInitials[[1]]$initialXreg %*% xregData[1,]);
                                }
                                else{
                                    matVt[1,1:lagsModelMax] <- matVt[1,lagsModelMax] /
                                        as.vector(exp(xregModelInitials[[2]]$initialXreg %*% xregData[1,]));
                                }
                            }
                        }
                        else{
                            matVt[1,1:lagsModelMax] <- initialLevel;
                        }
                        if(modelIsTrendy){
                            if(initialTrendEstimate){
                                matVt[2,1:lagsModelMax] <- switch(Ttype,
                                                                  "A" = mean(diff(yInSample[1:max(lagsModelMax+1,
                                                                                                  ceiling(obsInSample*0.2))]),
                                                                             na.rm=TRUE),
                                                                  "M" = exp(mean(diff(log(yInSample[otLogical])),na.rm=TRUE)));
                            }
                            else{
                                matVt[2,1:lagsModelMax] <- initialTrend;
                            }
                        }
                    }

                    if(initialLevelEstimate && Etype=="M" && matVt[1,lagsModelMax]==0){
                        matVt[1,1:lagsModelMax] <- mean(yInSample);
                    }
                }
                # Else, insert the provided ones... make sure that this is not a backcasting
                else if(!initialEstimate && initialType=="provided"){
                    j <- 1;
                    matVt[j,1:lagsModelMax] <- initialLevel;
                    if(modelIsTrendy){
                        j <- j+1;
                        matVt[j,1:lagsModelMax] <- initialTrend;
                    }
                    if(modelIsSeasonal){
                        for(i in 1:componentsNumberETSSeasonal){
                            matVt[j+i,(lagsModelMax-lagsModel[j+i])+1:lagsModel[j+i]] <- initialSeasonal[[i]];
                        }
                    }
                    j <- j+componentsNumberETSSeasonal;
                }
            }

            # If ARIMA orders are specified, prepare initials
            if(arimaModel){
                if(initialArimaEstimate){
                    matVt[componentsNumberETS+1:componentsNumberARIMA, 1:lagsModelARIMA[componentsNumberARIMA]] <-
                        switch(Etype, "A"=0, "M"=1);

                    # If this is just ARIMA with optimisation, refine the initials
                    if(!etsModel && initialType!="backcasting"){
                        # This is needed in order to make the initial components more realistic
                        # matVt[1:componentsNumberARIMA,
                        #       1:lagsModelARIMA[componentsNumberARIMA]+(lagsModelMax-lagsModelARIMA[componentsNumberARIMA])] <-
                        #     yInSample[lagsModelMax:1];

                        arimaPolynomials <- polynomialiser(rep(0.1,sum(c(arOrders,maOrders))), arOrders, iOrders, maOrders,
                                                           arRequired, maRequired, arEstimate, maEstimate, armaParameters, lags);
                        if(nrow(nonZeroARI)>0 && nrow(nonZeroARI)>=nrow(nonZeroMA)){
                            matVt[componentsNumberETS+nonZeroARI[,2],
                                  1:initialArimaNumber] <-
                                switch(Etype,
                                       "A"=arimaPolynomials$ariPolynomial[nonZeroARI[,1]] %*%
                                           t(matVt[componentsNumberARIMA, 1:initialArimaNumber]) /
                                           tail(arimaPolynomials$ariPolynomial,1),
                                       "M"=exp(arimaPolynomials$ariPolynomial[nonZeroARI[,1]] %*%
                                                   t(log(matVt[componentsNumberARIMA, 1:initialArimaNumber])) /
                                                   tail(arimaPolynomials$ariPolynomial,1)));
                        }
                        else{
                            matVt[componentsNumberETS+nonZeroMA[,2],
                                  1:initialArimaNumber] <-
                                switch(Etype,
                                       "A"=arimaPolynomials$maPolynomial[nonZeroMA[,1]] %*%
                                           t(matVt[componentsNumberARIMA, 1:initialArimaNumber]) /
                                           tail(arimaPolynomials$maPolynomial,1),
                                       "M"=exp(arimaPolynomials$maPolynomial[nonZeroMA[,1]] %*%
                                                   t(log(matVt[componentsNumberARIMA, 1:initialArimaNumber])) /
                                                   tail(arimaPolynomials$maPolynomial,1)));
                        }
                    }
                }
                else{
                    # Fill in the matrix with 0 / 1, just in case if the state will not be updated anymore
                    matVt[componentsNumberETS+1:componentsNumberARIMA, 1:lagsModelARIMA[componentsNumberARIMA]] <-
                        switch(Etype, "A"=0, "M"=1);
                    # Insert the provided initials
                    matVt[componentsNumberETS+componentsNumberARIMA, 1:lagsModelARIMA[componentsNumberARIMA]] <-
                        initialArima[1:initialArimaNumber];

                    # If only AR is needed, but provided or if both are needed, but provided
                    if(((arRequired && !arEstimate) && !maRequired) ||
                       ((arRequired && !arEstimate) && (maRequired && !maEstimate)) ||
                       (iRequired && !arEstimate && !maEstimate)){
                        matVt[componentsNumberETS+nonZeroARI[,2],1:initialArimaNumber] <-
                            switch(Etype,
                                   "A"=arimaPolynomials$ariPolynomial[nonZeroARI[,1]] %*%
                                       t(initialArima[1:initialArimaNumber]) /
                                       tail(arimaPolynomials$ariPolynomial,1),
                                   "M"=exp(arimaPolynomials$ariPolynomial[nonZeroARI[,1]] %*%
                                               t(log(initialArima[1:initialArimaNumber])) /
                                               tail(arimaPolynomials$ariPolynomial,1)));
                    }
                    # If only MA is needed, but provided
                    else if(((maRequired && !maEstimate) && !arRequired)){
                        matVt[componentsNumberETS+nonZeroMA[,2],1:initialArimaNumber] <-
                            switch(Etype,
                                   "A"=arimaPolynomials$maPolynomial[nonZeroMA[,1]] %*%
                                       t(initialArima[1:initialArimaNumber]) /
                                       tail(arimaPolynomials$maPolynomial,1),
                                   "M"=exp(arimaPolynomials$maPolynomial[nonZeroMA[,1]] %*%
                                               t(log(initialArima[1:initialArimaNumber])) /
                                               tail(arimaPolynomials$maPolynomial,1)));
                    }
                }
            }

            # Fill in the initials for xreg
            if(xregModel){
                if(Etype=="A" || initialXregProvided || is.null(xregModelInitials[[2]])){
                    matVt[componentsNumberETS+componentsNumberARIMA+1:xregNumber,
                          1:lagsModelMax] <- xregModelInitials[[1]]$initialXreg;
                }
                else{
                    matVt[componentsNumberETS+componentsNumberARIMA+1:xregNumber,
                          1:lagsModelMax] <- xregModelInitials[[2]]$initialXreg;
                }
            }
        }
        else{
            matVt[,1:lagsModelMax] <- profilesRecentTable;
        }

        return(list(matVt=matVt, matWt=matWt, matF=matF, vecG=vecG, arimaPolynomials=arimaPolynomials));
    }

    #### The function fills in the existing matrices with values of A ####
    # This is needed in order to do the estimation and the fit
    filler <- function(B,
                       etsModel, Etype, Ttype, Stype, modelIsTrendy, modelIsSeasonal,
                       componentsNumberETS, componentsNumberETSNonSeasonal,
                       componentsNumberETSSeasonal, componentsNumberARIMA,
                       lags, lagsModel, lagsModelMax,
                       matVt, matWt, matF, vecG,
                       persistenceEstimate, persistenceLevelEstimate, persistenceTrendEstimate,
                       persistenceSeasonalEstimate, persistenceXregEstimate,
                       phiEstimate,
                       initialType, initialEstimate,
                       initialLevelEstimate, initialTrendEstimate, initialSeasonalEstimate,
                       initialArimaEstimate, initialXregEstimate,
                       arimaModel, arEstimate, maEstimate, arOrders, iOrders, maOrders,
                       arRequired, maRequired, armaParameters,
                       nonZeroARI, nonZeroMA, arimaPolynomials,
                       xregModel, xregNumber,
                       xregParametersMissing, xregParametersIncluded,
                       xregParametersEstimated, xregParametersPersistence){

        j <- 0;
        # Fill in persistence
        if(persistenceEstimate){
            # Persistence of ETS
            if(etsModel){
                i <- 1;
                # alpha
                if(persistenceLevelEstimate){
                    j[] <- j+1;
                    vecG[i] <- B[j];
                }
                # beta
                if(modelIsTrendy){
                    i[] <- 2;
                    if(persistenceTrendEstimate){
                        j[] <- j+1;
                        vecG[i] <- B[j];
                    }
                }
                # gamma1, gamma2, ...
                if(modelIsSeasonal){
                    if(any(persistenceSeasonalEstimate)){
                        vecG[i+which(persistenceSeasonalEstimate)] <- B[j+c(1:sum(persistenceSeasonalEstimate))];
                        j[] <- j+sum(persistenceSeasonalEstimate);
                    }
                    i[] <- componentsNumberETS;
                }
            }

            # Persistence of xreg
            if(xregModel && persistenceXregEstimate){
                xregPersistenceNumber <- max(xregParametersPersistence);
                vecG[j+componentsNumberARIMA+1:length(xregParametersPersistence)] <-
                    B[j+1:xregPersistenceNumber][xregParametersPersistence];
                j[] <- j+xregPersistenceNumber;
            }
        }

        # Damping parameter
        if(etsModel && phiEstimate){
            j[] <- j+1;
            matWt[,2] <- B[j];
            matF[1:2,2] <- B[j];
        }

        # ARMA parameters. This goes before xreg in persistence
        if(arimaModel){
            # Call the function returning ARI and MA polynomials
            arimaPolynomials <- polynomialiser(B[j+1:sum(c(arOrders*arEstimate,maOrders*maEstimate))], arOrders, iOrders, maOrders,
                                               arRequired, maRequired, arEstimate, maEstimate, armaParameters, lags);

            # Fill in the transition matrix
            if(nrow(nonZeroARI)>0){
                matF[componentsNumberETS+nonZeroARI[,2],componentsNumberETS+1:componentsNumberARIMA] <-
                    -arimaPolynomials$ariPolynomial[nonZeroARI[,1]];
            }
            # Fill in the persistence vector
            if(nrow(nonZeroARI)>0){
                vecG[componentsNumberETS+nonZeroARI[,2]] <- -arimaPolynomials$ariPolynomial[nonZeroARI[,1]];
            }
            if(nrow(nonZeroMA)>0){
                vecG[componentsNumberETS+nonZeroMA[,2]] <- vecG[componentsNumberETS+nonZeroMA[,2]] +
                    arimaPolynomials$maPolynomial[nonZeroMA[,1]];
            }
            j[] <- j+sum(c(arOrders*arEstimate,maOrders*maEstimate));
        }

        # Initials of ETS if something needs to be estimated
        if(etsModel && (initialType!="backcasting") && initialEstimate){
            i <- 1;
            if(initialLevelEstimate){
                j[] <- j+1;
                matVt[i,1:lagsModelMax] <- B[j];
            }
            i[] <- i+1;
            if(modelIsTrendy && initialTrendEstimate){
                j[] <- j+1;
                matVt[i,1:lagsModelMax] <- B[j];
                i[] <- i+1;
            }
            if(modelIsSeasonal && any(initialSeasonalEstimate)){
                for(k in 1:componentsNumberETSSeasonal){
                    if(initialSeasonalEstimate[k]){
                        matVt[componentsNumberETSNonSeasonal+k, 2:lagsModel[componentsNumberETSNonSeasonal+k]-1] <-
                            B[j+2:(lagsModel[componentsNumberETSNonSeasonal+k])-1];
                        matVt[componentsNumberETSNonSeasonal+k, lagsModel[componentsNumberETSNonSeasonal+k]] <-
                            switch(Stype,
                                   "A"=-sum(B[j+2:(lagsModel[componentsNumberETSNonSeasonal+k])-1]),
                                   "M"=1/prod(B[j+2:(lagsModel[componentsNumberETSNonSeasonal+k])-1]));
                        j[] <- j+lagsModel[componentsNumberETSNonSeasonal+k]-1;
                    }
                }
            }
        }

        # Initials of ARIMA
        if(arimaModel){
            if((initialType!="backcasting") && initialArimaEstimate){
                if(nrow(nonZeroARI)>0 && nrow(nonZeroARI)>=nrow(nonZeroMA)){
                    matVt[componentsNumberETS+componentsNumberARIMA, 1:initialArimaNumber] <- B[j+1:initialArimaNumber];
                    matVt[componentsNumberETS+nonZeroARI[,2], 1:initialArimaNumber] <-
                        switch(Etype,
                               "A"=arimaPolynomials$ariPolynomial[nonZeroARI[,1]] %*% t(B[j+1:initialArimaNumber]) /
                                   tail(arimaPolynomials$ariPolynomial,1),
                               "M"=exp(arimaPolynomials$ariPolynomial[nonZeroARI[,1]] %*% t(log(B[j+1:initialArimaNumber])) /
                                           tail(arimaPolynomials$ariPolynomial,1)));
                }
                else{
                    matVt[componentsNumberETS+componentsNumberARIMA, 1:initialArimaNumber] <- B[j+1:initialArimaNumber];
                    matVt[componentsNumberETS+nonZeroMA[,2], 1:initialArimaNumber] <-
                        switch(Etype,
                               "A"=arimaPolynomials$maPolynomial[nonZeroMA[,1]] %*% t(B[j+1:initialArimaNumber]) /
                                   tail(arimaPolynomials$maPolynomial,1),
                               "M"=exp(arimaPolynomials$maPolynomial[nonZeroMA[,1]] %*% t(log(B[j+1:initialArimaNumber])) /
                                           tail(arimaPolynomials$maPolynomial,1)));
                }
                j[] <- j+componentsNumberARIMA;
            }
            # This is needed in order to propagate initials of ARIMA to all components
            else if(any(c(arEstimate,maEstimate))){
                if(nrow(nonZeroARI)>0 && nrow(nonZeroARI)>=nrow(nonZeroMA)){
                    matVt[componentsNumberETS+nonZeroARI[,2], 1:initialArimaNumber] <-
                        switch(Etype,
                               "A"= arimaPolynomials$ariPolynomial[nonZeroARI[,1]] %*%
                                   t(matVt[componentsNumberETS+componentsNumberARIMA, 1:initialArimaNumber]) /
                                   tail(arimaPolynomials$ariPolynomial,1),
                               "M"=exp(arimaPolynomials$ariPolynomial[nonZeroARI[,1]] %*%
                                           t(log(matVt[componentsNumberETS+componentsNumberARIMA, 1:initialArimaNumber])) /
                                           tail(arimaPolynomials$ariPolynomial,1)));
                }
                else{
                    matVt[componentsNumberETS+nonZeroMA[,2],
                          1:initialArimaNumber] <-
                        switch(Etype,
                               "A"=arimaPolynomials$maPolynomial[nonZeroMA[,1]] %*%
                                   t(matVt[componentsNumberETS+componentsNumberARIMA, 1:initialArimaNumber]) /
                                   tail(arimaPolynomials$maPolynomial,1),
                               "M"=exp(arimaPolynomials$maPolynomial[nonZeroMA[,1]] %*%
                                           t(log(matVt[componentsNumberETS+componentsNumberARIMA, 1:initialArimaNumber])) /
                                           tail(arimaPolynomials$maPolynomial,1)));
                }
            }
        }

        # Initials of the xreg
        if(xregModel && (initialType!="backcasting") && initialEstimate && initialXregEstimate){
            xregNumberToEstimate <- sum(xregParametersEstimated);
            matVt[componentsNumberETS+componentsNumberARIMA+which(xregParametersEstimated==1),
                  1:lagsModelMax] <- B[j+1:xregNumberToEstimate];
            # Normalise initials
            for(i in which(xregParametersMissing!=0)){
                matVt[componentsNumberETS+componentsNumberARIMA+i,
                      1:lagsModelMax] <- -sum(matVt[componentsNumberETS+componentsNumberARIMA+
                                                        which(xregParametersIncluded==xregParametersMissing[i]),
                                                    1:lagsModelMax]);
            }
        }

        return(list(matVt=matVt, matWt=matWt, matF=matF, vecG=vecG, arimaPolynomials=arimaPolynomials));
    }

    #### The function initialises the vector B for ETS ####
    initialiser <- function(etsModel, Etype, Ttype, Stype, modelIsTrendy, modelIsSeasonal,
                            componentsNumberETSNonSeasonal, componentsNumberETSSeasonal, componentsNumberETS,
                            lags, lagsModel, lagsModelSeasonal, lagsModelARIMA, lagsModelMax,
                            matVt,
                            # persistence values
                            persistenceEstimate, persistenceLevelEstimate, persistenceTrendEstimate,
                            persistenceSeasonalEstimate, persistenceXregEstimate,
                            # initials
                            phiEstimate, initialType, initialEstimate,
                            initialLevelEstimate, initialTrendEstimate, initialSeasonalEstimate,
                            initialArimaEstimate, initialXregEstimate,
                            # ARIMA elements
                            arimaModel, arRequired, maRequired, arEstimate, maEstimate, arOrders, maOrders,
                            componentsNumberARIMA, componentsNamesARIMA, initialArimaNumber,
                            # Explanatory variables
                            xregModel, xregNumber,
                            xregParametersEstimated, xregParametersPersistence,
                            otherParameterEstimate){
        # The vector of logicals for persistence elements
        persistenceEstimateVector <- c(persistenceLevelEstimate,modelIsTrendy&persistenceTrendEstimate,
                                       modelIsSeasonal&persistenceSeasonalEstimate);

        # The order:
        # Persistence of states and for xreg, phi, AR and MA parameters, initials, initialsARIMA, initials for xreg
        B <- Bl <- Bu <- vector("numeric",
                                # Values of the persistence vector + phi
                                etsModel*(persistenceLevelEstimate + modelIsTrendy*persistenceTrendEstimate +
                                              modelIsSeasonal*sum(persistenceSeasonalEstimate) + phiEstimate) +
                                    xregModel*persistenceXregEstimate*max(xregParametersPersistence) +
                                    # AR and MA values
                                    arimaModel*(arEstimate*sum(arOrders)+maEstimate*sum(maOrders)) +
                                    # initials of ETS
                                    etsModel*(initialType!="backcasting")*
                                    (initialLevelEstimate +
                                         (modelIsTrendy*initialTrendEstimate) +
                                         (modelIsSeasonal*sum(initialSeasonalEstimate*(lagsModelSeasonal-1)))) +
                                    # initials of ARIMA
                                    (initialType!="backcasting")*arimaModel*initialArimaNumber*initialArimaEstimate +
                                    # initials of xreg
                                    (initialType!="backcasting")*xregModel*initialXregEstimate*sum(xregParametersEstimated)+
                                    otherParameterEstimate);

        j <- 0;
        if(etsModel){
            # Fill in persistence
            if(persistenceEstimate){
                if(any(c(Etype,Ttype,Stype)=="M")){
                    # A special type of model which is not safe: AAM, MAA, MAM
                    if((Etype=="A" && Ttype=="A" && Stype=="M") || (Etype=="A" && Ttype=="M" && Stype=="A") ||
                       ((initialType=="backcasting") &&
                        ((Etype=="M" && Ttype=="A" && Stype=="A") || (Etype=="M" && Ttype=="A" && Stype=="M")))){
                        B[1:sum(persistenceEstimateVector)] <-
                            c(0.01,0,rep(0,componentsNumberETSSeasonal))[which(persistenceEstimateVector)];
                    }
                    # MMA is the worst. Set everything to zero and see if anything can be done...
                    else if((Etype=="M" && Ttype=="M" && Stype=="A")){
                        B[1:sum(persistenceEstimateVector)] <-
                            c(0,0,rep(0,componentsNumberETSSeasonal))[which(persistenceEstimateVector)];
                    }
                    else if(Etype=="M" && Ttype=="A"){
                        if(initialType=="backcasting"){
                            B[1:sum(persistenceEstimateVector)] <-
                                c(0.1,0,rep(0.01,componentsNumberETSSeasonal))[which(persistenceEstimateVector)];
                        }
                        else{
                            B[1:sum(persistenceEstimateVector)] <-
                                c(0.2,0.01,rep(0.01,componentsNumberETSSeasonal))[which(persistenceEstimateVector)];
                        }
                    }
                    else if(Etype=="M" && Ttype=="M"){
                        B[1:sum(persistenceEstimateVector)] <-
                            c(0.1,0.05,rep(0.01,componentsNumberETSSeasonal))[which(persistenceEstimateVector)];
                    }
                    else{
                        B[1:sum(persistenceEstimateVector)] <-
                            c(0.1,0.05,rep(0.05,componentsNumberETSSeasonal))[which(persistenceEstimateVector)];
                    }
                }
                else{
                    B[1:sum(persistenceEstimateVector)] <-
                        c(0.1,0.05,rep(0.11,componentsNumberETSSeasonal))[which(persistenceEstimateVector)];
                }
                Bl[1:sum(persistenceEstimateVector)] <- rep(-5, sum(persistenceEstimateVector));
                Bu[1:sum(persistenceEstimateVector)] <- rep(5, sum(persistenceEstimateVector));
                # Names for B
                if(persistenceLevelEstimate){
                    j[] <- j+1
                    names(B)[j] <- "alpha";
                }
                if(modelIsTrendy && persistenceTrendEstimate){
                    j[] <- j+1
                    names(B)[j] <- "beta";
                }
                if(modelIsSeasonal && any(persistenceSeasonalEstimate)){
                    if(componentsNumberETSSeasonal>1){
                        names(B)[j+c(1:sum(persistenceSeasonalEstimate))] <-
                            paste0("gamma",c(1:componentsNumberETSSeasonal));
                    }
                    else{
                        names(B)[j+1] <- "gamma";
                    }
                    j[] <- j+sum(persistenceSeasonalEstimate);
                }
            }
        }

        # Persistence if xreg is provided
        if(xregModel && persistenceXregEstimate){
            xregPersistenceNumber <- max(xregParametersPersistence);
            B[j+1:xregPersistenceNumber] <- rep(switch(Etype,"A"=0.01,"M"=0),xregPersistenceNumber);
            Bl[j+1:xregPersistenceNumber] <- rep(-5, xregPersistenceNumber);
            Bu[j+1:xregPersistenceNumber] <- rep(5, xregPersistenceNumber);
            names(B)[j+1:xregPersistenceNumber] <- paste0("delta",c(1:xregPersistenceNumber));
            j[] <- j+xregPersistenceNumber;
        }

        # Damping parameter
        if(etsModel && phiEstimate){
            j[] <- j+1;
            B[j] <- 0.95;
            names(B)[j] <- "phi";
            Bl[j] <- 0;
            Bu[j] <- 1;
        }

        # ARIMA parameters (AR / MA)
        if(arimaModel){
            # These are filled in lags-wise
            if(any(c(arEstimate,maEstimate))){
                for(i in 1:length(lags)){
                    if(arRequired && arEstimate && arOrders[i]>0){
                        B[j+c(1:arOrders[i])] <- rep(0.1,arOrders[i]);
                        Bl[j+c(1:arOrders[i])] <- -5;
                        Bu[j+c(1:arOrders[i])] <- 5;
                        names(B)[j+1:arOrders[i]] <- paste0("phi",1:arOrders[i],"[",lags[i],"]");
                        j[] <- j + arOrders[i];
                    }
                    if(maRequired && maEstimate && maOrders[i]>0){
                        B[j+c(1:maOrders[i])] <- rep(0.1,maOrders[i]);
                        Bl[j+c(1:maOrders[i])] <- -5;
                        Bu[j+c(1:maOrders[i])] <- 5;
                        names(B)[j+1:maOrders[i]] <- paste0("theta",1:maOrders[i],"[",lags[i],"]");
                        j[] <- j + maOrders[i];
                    }
                }
            }
        }

        # Initials
        if(etsModel && initialType!="backcasting" && initialEstimate){
            if(initialLevelEstimate){
                j[] <- j+1;
                B[j] <- matVt[1,1];
                names(B)[j] <- "level";
                if(Etype=="A"){
                    Bl[j] <- -Inf;
                    Bu[j] <- Inf;
                }
                else{
                    Bl[j] <- 0;
                    Bu[j] <- Inf;
                }
            }
            if(modelIsTrendy && initialTrendEstimate){
                j[] <- j+1;
                B[j] <- matVt[2,1];
                names(B)[j] <- "trend";
                if(Ttype=="A"){
                    Bl[j] <- -Inf;
                    Bu[j] <- Inf;
                }
                else{
                    Bl[j] <- 0;
                    Bu[j] <- Inf;
                }
            }
            if(modelIsSeasonal && any(initialSeasonalEstimate)){
                if(componentsNumberETSSeasonal>1){
                    for(k in 1:componentsNumberETSSeasonal){
                        if(initialSeasonalEstimate[k]){
                            # -1 is needed in order to remove the redundant seasonal element (normalisation)
                            B[j+2:lagsModel[componentsNumberETSNonSeasonal+k]-1] <-
                                matVt[componentsNumberETSNonSeasonal+k, 2:lagsModel[componentsNumberETSNonSeasonal+k]-1];
                            names(B)[j+2:(lagsModel[componentsNumberETSNonSeasonal+k])-1] <-
                                paste0("seasonal",k,"_",2:lagsModel[componentsNumberETSNonSeasonal+k]-1);
                            if(Stype=="A"){
                                Bl[j+2:lagsModel[componentsNumberETSNonSeasonal+k]-1] <- -Inf;
                                Bu[j+2:lagsModel[componentsNumberETSNonSeasonal+k]-1] <- Inf;
                            }
                            else{
                                Bl[j+2:lagsModel[componentsNumberETSNonSeasonal+k]-1] <- 0;
                                Bu[j+2:lagsModel[componentsNumberETSNonSeasonal+k]-1] <- Inf;
                            }
                            j[] <- j+(lagsModelSeasonal[k]-1);
                        }
                    }
                }
                else{
                    # -1 is needed in order to remove the redundant seasonal element (normalisation)
                    B[j+2:(lagsModel[componentsNumberETS])-1] <- matVt[componentsNumberETS,2:lagsModel[componentsNumberETS]-1];
                    names(B)[j+2:(lagsModel[componentsNumberETS])-1] <- paste0("seasonal_",2:lagsModel[componentsNumberETS]-1);
                    if(Stype=="A"){
                        Bl[j+2:(lagsModel[componentsNumberETS])-1] <- -Inf;
                        Bu[j+2:(lagsModel[componentsNumberETS])-1] <- Inf;
                    }
                    else{
                        Bl[j+2:(lagsModel[componentsNumberETS])-1] <- 0;
                        Bu[j+2:(lagsModel[componentsNumberETS])-1] <- Inf;
                    }
                    j[] <- j+(lagsModel[componentsNumberETS]-1);
                }
            }
        }

        # ARIMA initials
        if(initialType!="backcasting" && arimaModel && initialArimaEstimate){
            B[j+1:initialArimaNumber] <- head(matVt[componentsNumberETS+componentsNumberARIMA,1:lagsModelMax],initialArimaNumber);
            names(B)[j+1:initialArimaNumber] <- paste0("ARIMAState",1:initialArimaNumber);
            j[] <- j+initialArimaNumber;
        }

        # Initials of the xreg
        if(initialType!="backcasting" && initialXregEstimate){
            xregNumberToEstimate <- sum(xregParametersEstimated);
            B[j+1:xregNumberToEstimate] <- matVt[componentsNumberETS+componentsNumberARIMA+
                                                     which(xregParametersEstimated==1),1];
            names(B)[j+1:xregNumberToEstimate] <- rownames(matVt)[componentsNumberETS+componentsNumberARIMA+
                                                                      which(xregParametersEstimated==1)];
            if(Etype=="A"){
                Bl[j+1:xregNumberToEstimate] <- -Inf;
                Bu[j+1:xregNumberToEstimate] <- Inf;
            }
            else{
                Bl[j+1:xregNumberToEstimate] <- -Inf;
                Bu[j+1:xregNumberToEstimate] <- Inf;
            }
            j[] <- j+xregNumberToEstimate;
        }

        # Add lambda if it is needed
        if(otherParameterEstimate){
            j[] <- j+1;
            B[j] <- other;
            names(B)[j] <- "other";
            Bl[j] <- 1e-10;
            Bu[j] <- Inf;
        }

        return(list(B=B,Bl=Bl,Bu=Bu));
    }

    ##### Function returns scale parameter for the provided parameters #####
    scaler <- function(distribution, Etype, errors, yFitted, obsInSample, other){
        # as.complex() is needed in order to make the optimiser work in exotic cases
        return(switch(distribution,
                      "dnorm"=sqrt(sum(errors^2)/obsInSample),
                      "dlaplace"=sum(abs(errors))/obsInSample,
                      "ds"=sum(sqrt(abs(errors))) / (obsInSample*2),
                      "dgnorm"=(other*sum(abs(errors)^other)/obsInSample)^{1/other},
                      "dlogis"=sqrt(sum(errors^2)/obsInSample * 3 / pi^2),
                      "dt"=sqrt(sum(errors^2)/obsInSample),
                      "dalaplace"=sum(errors*(other-(errors<=0)*1))/obsInSample,
                      # This condition guarantees that E(1+e_t)=1
                      # abs is needed for cases, when we get imaginary values - a failsafe
                      "dlnorm"=sqrt(2*abs(switch(Etype,
                                                 "A"=1-sqrt(abs(1-sum(log(abs(1+errors/yFitted))^2)/obsInSample)),
                                                 "M"=1-sqrt(abs(1-sum(log(1+errors)^2)/obsInSample))))),
                      # "A"=Re(sqrt(sum(log(as.complex(1+errors/yFitted))^2)/obsInSample)),
                      # "M"=sqrt(sum(log(1+errors)^2)/obsInSample)),
                      "dllaplace"=switch(Etype,
                                         "A"=Re(sum(abs(log(as.complex(1+errors/yFitted))))/obsInSample),
                                         "M"=sum(abs(log(1+errors))/obsInSample)),
                      "dls"=switch(Etype,
                                   "A"=Re(sum(sqrt(abs(log(as.complex(1+errors/yFitted))))/obsInSample)),
                                   "M"=sum(sqrt(abs(log(1+errors)))/obsInSample)),
                      "dlgnorm"=switch(Etype,
                                       "A"=Re((other*sum(abs(log(as.complex(1+errors/yFitted)))^other)/obsInSample)^{1/other}),
                                       "M"=(other*sum(abs(log(as.complex(1+errors)))^other)/obsInSample)^{1/other}),
                      "dinvgauss"=switch(Etype,
                                         "A"=sum((errors/yFitted)^2/(1+errors/yFitted))/obsInSample,
                                         "M"=sum((errors)^2/(1+errors))/obsInSample)
                      # "M"=mean((errors)^2/(1+errors))),
        ));
    }

    #### The function inverts the measurement matrix, setting infinte values to zero
    # This is needed for the stability check for xreg models with regressors="adapt"
    measurementInverter <- function(measurement){
        measurement[] <- 1/measurement;
        measurement[is.infinite(measurement)] <- 0;
        return(measurement);
    }

    ##### Cost Function for ETS #####
    CF <- function(B,
                   etsModel, Etype, Ttype, Stype, modelIsTrendy, modelIsSeasonal, yInSample,
                   ot, otLogical, occurrenceModel, obsInSample,
                   componentsNumberETS, componentsNumberETSSeasonal, componentsNumberETSNonSeasonal,
                   componentsNumberARIMA,
                   lags, lagsModel, lagsModelAll, lagsModelMax,
                   profilesObservedTable, profilesRecentTable,
                   matVt, matWt, matF, vecG,
                   persistenceEstimate, persistenceLevelEstimate, persistenceTrendEstimate,
                   persistenceSeasonalEstimate, persistenceXregEstimate,
                   phiEstimate, initialType, initialEstimate,
                   initialLevelEstimate, initialTrendEstimate, initialSeasonalEstimate,
                   initialArimaEstimate, initialXregEstimate,
                   arimaModel, nonZeroARI, nonZeroMA, arEstimate, maEstimate, arimaPolynomials,
                   arOrders, iOrders, maOrders, arRequired, maRequired, armaParameters,
                   xregModel, xregNumber,
                   xregParametersMissing, xregParametersIncluded,
                   xregParametersEstimated, xregParametersPersistence,
                   bounds, loss, lossFunction, distribution, horizon, multisteps, denominator=NULL,
                   other, otherParameterEstimate, lambda,
                   arPolynomialMatrix, maPolynomialMatrix){

        # Fill in the matrices
        adamElements <- filler(B,
                               etsModel, Etype, Ttype, Stype, modelIsTrendy, modelIsSeasonal,
                               componentsNumberETS, componentsNumberETSNonSeasonal,
                               componentsNumberETSSeasonal, componentsNumberARIMA,
                               lags, lagsModel, lagsModelMax,
                               matVt, matWt, matF, vecG,
                               persistenceEstimate, persistenceLevelEstimate, persistenceTrendEstimate,
                               persistenceSeasonalEstimate, persistenceXregEstimate,
                               phiEstimate,
                               initialType, initialEstimate,
                               initialLevelEstimate, initialTrendEstimate, initialSeasonalEstimate,
                               initialArimaEstimate, initialXregEstimate,
                               arimaModel, arEstimate, maEstimate, arOrders, iOrders, maOrders,
                               arRequired, maRequired, armaParameters,
                               nonZeroARI, nonZeroMA, arimaPolynomials,
                               xregModel, xregNumber,
                               xregParametersMissing, xregParametersIncluded,
                               xregParametersEstimated, xregParametersPersistence);

        # If we estimate parameters of distribution, take it from the B vector
        if(otherParameterEstimate){
            # Take absolute value, just to be on safe side. We don't need negatives anyway.
            other[] <- abs(B[length(B)]);

            # Beta in GN is restricted by 0.25 if it is optimised.
            if(any(distribution==c("dgnorm","dlgnorm")) && other<0.25){
                return(1E+10/other);
            }
        }

        # Check the bounds, classical restrictions
        #### The usual bounds ####
        if(bounds=="usual"){
            # Stationarity and invertibility conditions for ARIMA
            if(arimaModel && any(c(arEstimate,maEstimate))){
                # Calculate the polynomial roots for AR
                if(arEstimate){
                    arPolynomialMatrix[,1] <- -adamElements$arimaPolynomials$arPolynomial[-1];
                    arPolyroots <- abs(eigen(arPolynomialMatrix, symmetric=TRUE, only.values=TRUE)$values);
                    if(any(arPolyroots>1)){
                        return(1E+100*max(arPolyroots));
                    }
                }
                # Calculate the polynomial roots of MA
                if(maEstimate){
                    maPolynomialMatrix[,1] <- adamElements$arimaPolynomials$maPolynomial[-1];
                    maPolyroots <- abs(eigen(maPolynomialMatrix, symmetric=TRUE ,only.values=TRUE)$values);
                    if(any(maPolyroots>1)){
                        return(1E+100*max(abs(maPolyroots)));
                    }
                }
            }

            # Smoothing parameters & phi restrictions in case of ETS
            if(etsModel){
                if(any(adamElements$vecG[1:componentsNumberETS]>1) || any(adamElements$vecG[1:componentsNumberETS]<0)){
                    return(1E+300);
                }
                if(modelIsTrendy){
                    if((adamElements$vecG[2]>adamElements$vecG[1])){
                        return(1E+300);
                    }
                    if(modelIsSeasonal && any(adamElements$vecG[componentsNumberETSNonSeasonal+c(1:componentsNumberETSSeasonal)]>
                                              (1-adamElements$vecG[1]))){
                        return(1E+300);
                    }
                }
                else{
                    if(modelIsSeasonal && any(adamElements$vecG[componentsNumberETSNonSeasonal+c(1:componentsNumberETSSeasonal)]>
                                              (1-adamElements$vecG[1]))){
                        return(1E+300);
                    }
                }

                # This is the restriction on the damping parameter
                if(phiEstimate && (adamElements$matF[2,2]>1 || adamElements$matF[2,2]<0)){
                    return(1E+300);
                }
            }

            # Smoothing parameters for the explanatory variables (0, 1) region
            if(xregModel && regressors=="adapt"){
                if(any(adamElements$vecG[componentsNumberETS+componentsNumberARIMA+1:xregNumber]>1) ||
                   any(adamElements$vecG[componentsNumberETS+componentsNumberARIMA+1:xregNumber]<0)){
                    return(1E+100*max(abs(adamElements$vecG[componentsNumberETS+componentsNumberARIMA+1:xregNumber]-0.5)));
                }
            }
        }
        #### The admissible bounds ####
        else if(bounds=="admissible"){
            # Stationarity condition of ARIMA
            if(arimaModel){
                # Calculate the polynomial roots for AR
                if(arEstimate){
                    arPolynomialMatrix[,1] <- -adamElements$arimaPolynomials$arPolynomial[-1];
                    arPolyroots <- abs(eigen(arPolynomialMatrix, symmetric=TRUE, only.values=TRUE)$values);
                    if(any(arPolyroots>1)){
                        return(1E+100*max(arPolyroots));
                    }
                }
            }

            # Stability / invertibility condition for ETS / ARIMA.
            if(etsModel || arimaModel){
                if(xregModel){
                    if(regressors=="adapt"){
                        # We check the condition on average
                        eigenValues <- abs(eigen((adamElements$matF -
                                                      diag(as.vector(adamElements$vecG)) %*%
                                                      t(measurementInverter(adamElements$matWt[1:obsInSample,,drop=FALSE])) %*%
                                                      adamElements$matWt[1:obsInSample,,drop=FALSE] / obsInSample),
                                                 symmetric=TRUE, only.values=TRUE)$values);
                    }
                    else{
                        # We drop the X parts from matrices
                        indices <- c(1:(componentsNumberETS+componentsNumberARIMA))
                        eigenValues <- abs(eigen(adamElements$matF[indices,indices,drop=FALSE] -
                                                     adamElements$vecG[indices,,drop=FALSE] %*%
                                                     adamElements$matWt[obsInSample,indices,drop=FALSE],
                                                 symmetric=TRUE, only.values=TRUE)$values);
                    }
                }
                else{
                    eigenValues <- abs(eigen(adamElements$matF -
                                                 adamElements$vecG %*% adamElements$matWt[obsInSample,,drop=FALSE],
                                             symmetric=TRUE, only.values=TRUE)$values);
                }
                if(any(eigenValues>1+1E-50)){
                    return(1E+100*max(eigenValues));
                }
            }
        }

        # Write down the initials in the recent profile
        profilesRecentTable[] <- adamElements$matVt[,1:lagsModelMax];

        #### Fitter and the losss calculation ####
        adamFitted <- adamFitterWrap(adamElements$matVt, adamElements$matWt, adamElements$matF, adamElements$vecG,
                                     lagsModelAll, profilesObservedTable, profilesRecentTable,
                                     Etype, Ttype, Stype, componentsNumberETS, componentsNumberETSSeasonal,
                                     componentsNumberARIMA, xregNumber, yInSample, ot, initialType=="backcasting");

        if(!multisteps){
            if(loss=="likelihood"){
                # Scale for different functions
                scale <- scaler(distribution, Etype, adamFitted$errors[otLogical],
                                adamFitted$yFitted[otLogical], obsInSample, other);

                # Calculate the likelihood
                ## as.complex() is needed for failsafe in case of exotic models
                CFValue <- -sum(switch(distribution,
                                       "dnorm"=switch(Etype,
                                                      "A"=dnorm(x=yInSample[otLogical], mean=adamFitted$yFitted[otLogical],
                                                                sd=scale, log=TRUE),
                                                      "M"=dnorm(x=yInSample[otLogical], mean=adamFitted$yFitted[otLogical],
                                                                sd=scale*adamFitted$yFitted[otLogical], log=TRUE)),
                                       "dlaplace"=switch(Etype,
                                                         "A"=dlaplace(q=yInSample[otLogical], mu=adamFitted$yFitted[otLogical],
                                                                      scale=scale, log=TRUE),
                                                         "M"=dlaplace(q=yInSample[otLogical], mu=adamFitted$yFitted[otLogical],
                                                                      scale=scale*adamFitted$yFitted[otLogical], log=TRUE)),
                                       "ds"=switch(Etype,
                                                   "A"=ds(q=yInSample[otLogical],mu=adamFitted$yFitted[otLogical],
                                                          scale=scale, log=TRUE),
                                                   "M"=ds(q=yInSample[otLogical],mu=adamFitted$yFitted[otLogical],
                                                          scale=scale*sqrt(adamFitted$yFitted[otLogical]), log=TRUE)),
                                       "dgnorm"=switch(Etype,
                                                       "A"=dgnorm(x=yInSample[otLogical],mu=adamFitted$yFitted[otLogical],
                                                                  alpha=scale, beta=other, log=TRUE),
                                                       # Suppres Warnings is needed, because the check is done for scalar alpha
                                                       "M"=suppressWarnings(dgnorm(x=yInSample[otLogical],
                                                                                   mu=adamFitted$yFitted[otLogical],
                                                                                   alpha=scale*(adamFitted$yFitted[otLogical])^other,
                                                                                   beta=other, log=TRUE))),
                                       "dlogis"=switch(Etype,
                                                       "A"=dlogis(x=yInSample[otLogical],
                                                                  location=adamFitted$yFitted[otLogical],
                                                                  scale=scale, log=TRUE),
                                                       "M"=dlogis(x=yInSample[otLogical],
                                                                  location=adamFitted$yFitted[otLogical],
                                                                  scale=scale*adamFitted$yFitted[otLogical], log=TRUE)),
                                       "dt"=switch(Etype,
                                                   "A"=dt(adamFitted$errors[otLogical], df=abs(other), log=TRUE),
                                                   "M"=dt(adamFitted$errors[otLogical]*adamFitted$yFitted[otLogical],
                                                          df=abs(other), log=TRUE)),
                                       "dalaplace"=switch(Etype,
                                                          "A"=dalaplace(q=yInSample[otLogical],
                                                                        mu=adamFitted$yFitted[otLogical],
                                                                        scale=scale, alpha=other, log=TRUE),
                                                          "M"=dalaplace(q=yInSample[otLogical],
                                                                        mu=adamFitted$yFitted[otLogical],
                                                                        scale=scale*adamFitted$yFitted[otLogical],
                                                                        alpha=other, log=TRUE)),
                                       "dlnorm"=dlnorm(x=yInSample[otLogical],
                                                       meanlog=Re(log(as.complex(adamFitted$yFitted[otLogical])))-scale^2/2,
                                                       sdlog=scale, log=TRUE),
                                       "dllaplace"=dlaplace(q=log(yInSample[otLogical]),
                                                            mu=Re(log(as.complex(adamFitted$yFitted[otLogical]))),
                                                            scale=scale, log=TRUE) -log(yInSample[otLogical]),
                                       "dls"=ds(q=log(yInSample[otLogical]),
                                                mu=Re(log(as.complex(adamFitted$yFitted[otLogical]))),
                                                scale=scale, log=TRUE) -log(yInSample[otLogical]),
                                       "dlgnorm"=dgnorm(x=log(yInSample[otLogical]),
                                                        mu=Re(log(as.complex(adamFitted$yFitted[otLogical]))),
                                                        alpha=scale, beta=other, log=TRUE) -log(yInSample[otLogical]),
                                       # "dinvgauss"=dinvgauss(x=1+adamFitted$errors, mean=1,
                                       #                       dispersion=scale, log=TRUE)));
                                       # "dinvgauss"=dinvgauss(x=yInSampleNew, mean=adamFitted$yFitted,
                                       #                       dispersion=scale/adamFitted$yFitted, log=TRUE)));
                                       "dinvgauss"=dinvgauss(x=yInSample[otLogical], mean=adamFitted$yFitted[otLogical],
                                                             dispersion=scale/adamFitted$yFitted[otLogical], log=TRUE)));

                # Differential entropy for the logLik of occurrence model
                if(occurrenceModel || any(!otLogical)){
                    CFValue <- CFValue + switch(distribution,
                                                "dnorm" = obsZero*(log(sqrt(2*pi)*scale)+0.5),
                                                "dlnorm" = obsZero*(log(sqrt(2*pi)*scale)+0.5)-scale^2/2,
                                                "dlogis" = obsZero*2,
                                                "dlaplace" =,
                                                "dllaplace" =,
                                                "dalaplace" = obsZero*(1 + log(2*scale)),
                                                "ds" =,
                                                "dls" = obsZero*(2 + 2*log(2*scale)),
                                                "dgnorm" =,
                                                "dlgnorm" = obsZero*(1/other-log(other/(2*scale*gamma(1/other)))),
                                                "dt" = obsZero*((scale+1)/2 *
                                                                    (digamma((scale+1)/2)-digamma(scale/2)) +
                                                                    log(sqrt(scale) * beta(scale/2,0.5))),
                                                # "dinvgauss" = obsZero*(0.5*(log(pi/2)+1+suppressWarnings(log(scale)))));
                                                # "dinvgauss" =0);
                                                "dinvgauss" = 0.5*(obsZero*(log(pi/2)+1+suppressWarnings(log(scale)))-
                                                                       sum(log(adamFitted$yFitted[!otLogical]))));
                }
            }
            else if(loss=="MSE"){
                CFValue <- sum(adamFitted$errors^2)/obsInSample;
            }
            else if(loss=="MAE"){
                CFValue <- sum(abs(adamFitted$errors))/obsInSample;
            }
            else if(loss=="HAM"){
                CFValue <- sum(sqrt(abs(adamFitted$errors)))/obsInSample;
            }
            else if(any(loss==c("LASSO","RIDGE"))){
                ### All of this is needed in order to normalise level, trend, seasonal and xreg parameters
                # Define, how many elements to skip (we don't normalise smoothing parameters)
                if(persistenceXregEstimate){
                    persistenceToSkip <- componentsNumberETS+componentsNumberARIMA+xregNumber;
                }
                else{
                    persistenceToSkip <- componentsNumberETS+componentsNumberARIMA;
                }
                j <- 1;
                if(phiEstimate){
                    j[] <- 2;
                }
                if(initialType=="optimal"){
                    # Standardise the level
                    B[persistenceToSkip+j] <- (B[persistenceToSkip+j] - mean(yInSample[1:max(lagsModelMax,2)])) /
                        sd(yInSample[1:max(lagsModelMax,2)]);
                    # Change B values for the trend, so that it shrinks properly
                    if(Ttype=="M"){
                        j[] <- j+1;
                        B[persistenceToSkip+j] <- log(B[persistenceToSkip+j]);
                    }
                    else if(Ttype=="A"){
                        j[] <- j+1;
                        B[persistenceToSkip+j] <- B[persistenceToSkip+j] / sd(yInSample[1:max(lagsModelMax,2)]);
                    }
                    # Change B values for seasonality, so that it shrinks properly
                    if(Stype=="M"){
                        B[persistenceToSkip+j+1:(sum(lagsModelSeasonal)-componentsNumberETSSeasonal)] <-
                            log(B[persistenceToSkip+j+1:(sum(lagsModelSeasonal)-componentsNumberETSSeasonal)]);
                    }
                    else if(Stype=="A"){
                        B[persistenceToSkip+j+1:(sum(lagsModelSeasonal)-componentsNumberETSSeasonal)] <-
                            B[persistenceToSkip+j+1:(sum(lagsModelSeasonal)-componentsNumberETSSeasonal)] /
                            sd(yInSample[1:max(lagsModelMax,2)]);
                    }

                    # Normalise parameters of xreg if they are additive. Otherwise leave - they will be small and close to zero
                    if(xregNumber>0 && Etype=="A"){
                        B[persistenceToSkip+sum(lagsModel)+c(1:xregNumber)] <- tail(B,xregNumber) / denominator;
                    }
                }

                CFValue <- (switch(Etype,
                                   "A"=(1-lambda)* sqrt(sum(adamFitted$errors^2))/obsInSample,
                                   "M"=(1-lambda)* sqrt(sum(log(1+adamFitted$errors)^2))/obsInSample) +
                                switch(loss,
                                       "LASSO"=lambda * sum(abs(B)),
                                       "RIDGE"=lambda * sqrt(sum((B)^2))));
            }
            else if(loss=="custom"){
                CFValue <- lossFunction(actual=yInSample,fitted=adamFitted$yFitted,B=B);
            }
        }
        else{
            # Call for the Rcpp function to produce a matrix of multistep errors
            adamErrors <- adamErrorerWrap(adamFitted$matVt, adamElements$matWt, adamElements$matF,
                                          lagsModelAll, profilesObservedTable, profilesRecentTable,
                                          Etype, Ttype, Stype,
                                          componentsNumberETS, componentsNumberETSSeasonal,
                                          componentsNumberARIMA, xregNumber, h,
                                          yInSample, ot);

            # This is a fix for the multistep in case of Etype=="M", assuming logN
            if(Etype=="M"){
                adamErrors[] <- log(1+adamErrors);
            }

            # Not done yet: "aMSEh","aTMSE","aGTMSE","aMSCE","aGPL"
            CFValue <- switch(loss,
                              "MSEh"=sum(adamErrors[,h]^2)/(obsInSample-h),
                              "TMSE"=sum(colSums(adamErrors^2)/(obsInSample-h)),
                              "GTMSE"=sum(log(colSums(adamErrors^2)/(obsInSample-h))),
                              "MSCE"=sum(rowSums(adamErrors)^2)/(obsInSample-h),
                              "MAEh"=sum(abs(adamErrors[,h]))/(obsInSample-h),
                              "TMAE"=sum(colSums(abs(adamErrors))/(obsInSample-h)),
                              "GTMAE"=sum(log(colSums(abs(adamErrors))/(obsInSample-h))),
                              "MACE"=sum(abs(rowSums(adamErrors)))/(obsInSample-h),
                              "HAMh"=sum(sqrt(abs(adamErrors[,h])))/(obsInSample-h),
                              "THAM"=sum(colSums(sqrt(abs(adamErrors)))/(obsInSample-h)),
                              "GTHAM"=sum(log(colSums(sqrt(abs(adamErrors)))/(obsInSample-h))),
                              "CHAM"=sum(sqrt(abs(rowSums(adamErrors))))/(obsInSample-h),
                              "GPL"=log(det(t(adamErrors) %*% adamErrors/(obsInSample-h))),
                              0);

        }

        if(is.na(CFValue) || is.nan(CFValue)){
            CFValue[] <- 1e+300;
        }

        return(CFValue);
    }

    #### The function returns log-likelihood of the model ####
    logLikADAM <- function(B,
                           etsModel, Etype, Ttype, Stype, modelIsTrendy, modelIsSeasonal, yInSample,
                           ot, otLogical, occurrenceModel, pFitted, obsInSample,
                           componentsNumberETS, componentsNumberETSSeasonal, componentsNumberETSNonSeasonal,
                           componentsNumberARIMA,
                           lags, lagsModel, lagsModelAll, lagsModelMax,
                           profilesObservedTable, profilesRecentTable,
                           matVt, matWt, matF, vecG,
                           persistenceEstimate, persistenceLevelEstimate, persistenceTrendEstimate,
                           persistenceSeasonalEstimate, persistenceXregEstimate,
                           phiEstimate, initialType, initialEstimate,
                           initialLevelEstimate, initialTrendEstimate, initialSeasonalEstimate,
                           initialArimaEstimate, initialXregEstimate,
                           arimaModel, nonZeroARI, nonZeroMA, arEstimate, maEstimate, arimaPolynomials,
                           arOrders, iOrders, maOrders, arRequired, maRequired, armaParameters,
                           xregModel, xregNumber,
                           xregParametersMissing, xregParametersIncluded,
                           xregParametersEstimated, xregParametersPersistence,
                           bounds, loss, lossFunction, distribution, horizon, multisteps, denominator=NULL,
                           other, otherParameterEstimate, lambda,
                           arPolynomialMatrix, maPolynomialMatrix, hessianCalculation=FALSE){

        # If this is just for the calculation of hessian, return to the original values of parameters
        if(hessianCalculation && initialType=="optimal"){
            # Define, how many elements to skip (we don't normalise smoothing parameters)
            if(persistenceXregEstimate){
                persistenceToSkip <- componentsNumberETS+componentsNumberARIMA+xregNumber;
            }
            else{
                persistenceToSkip <- componentsNumberETS+componentsNumberARIMA;
            }
            j <- 1;
            if(phiEstimate){
                j[] <- 2;
            }
            # Level
            B[persistenceToSkip+j] <- B[persistenceToSkip+j] * sd(yInSample);
            # Trend
            if(Ttype!="N"){
                j[] <- j+1;
                if(Ttype=="A"){
                B[persistenceToSkip+j] <- B[persistenceToSkip+j] * sd(yInSample);
                }
            }
            # Seasonality
            if(Stype=="A"){
                if(componentsNumberETSSeasonal>1){
                    for(k in 1:componentsNumberETSSeasonal){
                        if(initialSeasonalEstimateFI[k]){
                            # -1 is needed in order to remove the redundant seasonal element (normalisation)
                            B[persistenceToSkip+j+2:lagsModel[componentsNumberETSNonSeasonal+k]-1] <-
                                B[persistenceToSkip+j+2:lagsModel[componentsNumberETSNonSeasonal+k]-1] *
                                sd(yInSample);
                            j[] <- j+(lagsModelSeasonal[k]-1);
                        }
                    }
                }
                else{
                    # -1 is needed in order to remove the redundant seasonal element (normalisation)
                    B[persistenceToSkip+j+2:(lagsModel[componentsNumberETS])-1] <-
                        B[persistenceToSkip+j+2:(lagsModel[componentsNumberETS])-1] * sd(yInSample);
                }
            }

            # Normalise parameters of xreg if they are additive. Otherwise leave - they will be small and close to zero
            if(xregNumber>0 && Etype=="A"){
                denominator <- tail(colMeans(abs(matWt)),xregNumber);
                # If it is lower than 1, then we are probably dealing with (0, 1). No need to normalise
                denominator[abs(denominator)<1] <- 1;
                B[persistenceToSkip+sum(lagsModel)+c(1:xregNumber)] <- tail(B,xregNumber) * denominator;
            }
        }

        if(!multisteps){
            if(any(loss==c("LASSO","RIDGE"))){
                return(0);
            }
            else{
                distributionNew <- switch(loss,
                                          "MSE"="dnorm",
                                          "MAE"="dlaplace",
                                          "HAM"="ds",
                                          distribution);

                lossNew <- switch(loss,
                                  "MSE"=,"MAE"=,"HAM"="likelihood",
                                  loss)

                # bounds="none" switches off the checks of parameters.
                logLikReturn <- -CF(B,
                                    etsModel, Etype, Ttype, Stype, modelIsTrendy, modelIsSeasonal, yInSample,
                                    ot, otLogical, occurrenceModel, obsInSample,
                                    componentsNumberETS, componentsNumberETSSeasonal, componentsNumberETSNonSeasonal,
                                    componentsNumberARIMA,
                                    lags, lagsModel, lagsModelAll, lagsModelMax,
                                    profilesObservedTable, profilesRecentTable,
                                    matVt, matWt, matF, vecG,
                                    persistenceEstimate, persistenceLevelEstimate, persistenceTrendEstimate,
                                    persistenceSeasonalEstimate, persistenceXregEstimate,
                                    phiEstimate, initialType, initialEstimate,
                                    initialLevelEstimate, initialTrendEstimate, initialSeasonalEstimate,
                                    initialArimaEstimate, initialXregEstimate,
                                    arimaModel, nonZeroARI, nonZeroMA, arEstimate, maEstimate, arimaPolynomials,
                                    arOrders, iOrders, maOrders, arRequired, maRequired, armaParameters,
                                    xregModel, xregNumber,
                                    xregParametersMissing, xregParametersIncluded,
                                    xregParametersEstimated, xregParametersPersistence,
                                    bounds="none", lossNew, lossFunction, distributionNew, horizon, multisteps,
                                    denominator, other, otherParameterEstimate, lambda,
                                    arPolynomialMatrix, maPolynomialMatrix);

                # If this is an occurrence model, add the probabilities
                if(occurrenceModel){
                    if(is.infinite(logLikReturn)){
                        logLikReturn[] <- 0;
                    }
                    if(any(c(1-pFitted[!otLogical]==0,pFitted[otLogical]==0))){
                        # return(-Inf);
                        ptNew <- pFitted[(pFitted!=0) & (pFitted!=1)];
                        otNew <- ot[(pFitted!=0) & (pFitted!=1)];

                        # Just return the original likelihood if the probability is weird
                        if(length(ptNew)==0){
                            return(logLikReturn);
                        }
                        else{
                            return(logLikReturn + sum(log(ptNew[otNew==1])) + sum(log(1-ptNew[otNew==0])));
                        }
                    }
                    else{
                        return(logLikReturn + sum(log(pFitted[otLogical])) + sum(log(1-pFitted[!otLogical])));
                    }
                }
                else{
                    return(logLikReturn);
                }
            }
        }
        else{
            # Use the predictive likelihoods from the GPL paper:
            # - Normal for MSEh, MSCE, GPL and their analytical counterparts
            # - Laplace for MAEh and MACE,
            # - S for HAMh and CHAM
            # bounds="none" switches off the checks of parameters.
            logLikReturn <- CF(B,
                               etsModel, Etype, Ttype, Stype, modelIsTrendy, modelIsSeasonal, yInSample,
                               ot, otLogical, occurrenceModel, obsInSample,
                               componentsNumberETS, componentsNumberETSSeasonal, componentsNumberETSNonSeasonal,
                               componentsNumberARIMA,
                               lags, lagsModel, lagsModelAll, lagsModelMax,
                               profilesObservedTable, profilesRecentTable,
                               matVt, matWt, matF, vecG,
                               persistenceEstimate, persistenceLevelEstimate, persistenceTrendEstimate,
                               persistenceSeasonalEstimate, persistenceXregEstimate,
                               phiEstimate, initialType, initialEstimate,
                               initialLevelEstimate, initialTrendEstimate, initialSeasonalEstimate,
                               initialArimaEstimate, initialXregEstimate,
                               arimaModel, nonZeroARI, nonZeroMA, arEstimate, maEstimate, arimaPolynomials,
                               arOrders, iOrders, maOrders, arRequired, maRequired, armaParameters,
                               xregModel, xregNumber,
                               xregParametersMissing, xregParametersIncluded,
                               xregParametersEstimated, xregParametersPersistence,
                               bounds="none", loss, lossFunction, distribution, horizon, multisteps,
                               denominator, other, otherParameterEstimate, lambda,
                               arPolynomialMatrix, maPolynomialMatrix);

            # Concentrated log-likelihoods for the multistep losses
            logLikReturn[] <- -switch(loss,
                                      "MSEh"=, "aMSEh"=, "TMSE"=, "aTMSE"=, "MSCE"=, "aMSCE"=
                                          (obsInSample-h)/2*(log(2*pi)+1+log(logLikReturn)),
                                      "GTMSE"=, "aGTMSE"=
                                          (obsInSample-h)/2*(log(2*pi)+1+logLikReturn),
                                      "MAEh"=, "TMAE"=, "GTMAE"=, "MACE"=
                                          (obsInSample-h)*(log(2)+1+log(logLikReturn)),
                                      "HAMh"=, "THAM"=, "GTHAM"=, "CHAM"=
                                          (obsInSample-h)*(log(4)+2+2*log(logLikReturn)),
                                      #### Divide GPL by 8 in order to make it comparable with the univariate ones
                                      "GPL"=, "aGPL"=
                                          (obsInSample-h)/2*(h*log(2*pi)+h+logLikReturn)/h);

            # This is not well motivated at the moment, but should make likelihood comparable, taking T instead of T-h
            logLikReturn[] <- logLikReturn / (obsInSample-h) * obsInSample;

            # In case of multiplicative model, we assume log- distribution
            if(Etype=="M"){
                logLikReturn[] <- logLikReturn - sum(log(yInSample));
            }

            return(logLikReturn);
        }
    }

    #### The function estimates the ETS model and returns B, logLik, nParam and CF(B) ####
    estimator <- function(etsModel, Etype, Ttype, Stype, lags, lagsModelSeasonal, lagsModelARIMA,
                          obsStates, obsInSample,
                          yInSample, persistence, persistenceEstimate,
                          persistenceLevel, persistenceLevelEstimate,
                          persistenceTrend, persistenceTrendEstimate,
                          persistenceSeasonal, persistenceSeasonalEstimate,
                          persistenceXreg, persistenceXregEstimate, persistenceXregProvided,
                          phi, phiEstimate,
                          initialType, initialLevel, initialTrend, initialSeasonal,
                          initialArima, initialEstimate,
                          initialLevelEstimate, initialTrendEstimate, initialSeasonalEstimate,
                          initialArimaEstimate, initialXregEstimate, initialXregProvided,
                          arimaModel, arRequired, iRequired, maRequired, armaParameters,
                          componentsNumberARIMA, componentsNamesARIMA,
                          formula, xregModel, xregModelInitials, xregData, xregNumber, xregNames, regressors,
                          xregParametersMissing, xregParametersIncluded,
                          xregParametersEstimated, xregParametersPersistence,
                          ot, otLogical, occurrenceModel, pFitted,
                          bounds, loss, lossFunction, distribution,
                          horizon, multisteps, other, otherParameterEstimate, lambda){

        # Create the basic variables
        adamArchitect <- architector(etsModel, Etype, Ttype, Stype, lags, lagsModelSeasonal,
                                     xregNumber, obsInSample, initialType,
                                     arimaModel, lagsModelARIMA, xregModel,
                                     profilesRecentTable, profilesRecentProvided);
        list2env(adamArchitect, environment());

        # Create the matrices for the specific ETS model
        adamCreated <- creator(etsModel, Etype, Ttype, Stype, modelIsTrendy, modelIsSeasonal,
                               lags, lagsModel, lagsModelARIMA, lagsModelAll, lagsModelMax,
                               profilesRecentTable, profilesRecentProvided,
                               obsStates, obsInSample, obsAll, componentsNumberETS, componentsNumberETSSeasonal,
                               componentsNamesETS, otLogical, yInSample,
                               persistence, persistenceEstimate,
                               persistenceLevel, persistenceLevelEstimate, persistenceTrend, persistenceTrendEstimate,
                               persistenceSeasonal, persistenceSeasonalEstimate,
                               persistenceXreg, persistenceXregEstimate, persistenceXregProvided,
                               phi,
                               initialType, initialEstimate,
                               initialLevel, initialLevelEstimate, initialTrend, initialTrendEstimate,
                               initialSeasonal, initialSeasonalEstimate,
                               initialArima, initialArimaEstimate, initialArimaNumber,
                               initialXregEstimate, initialXregProvided,
                               arimaModel, arRequired, iRequired, maRequired, armaParameters,
                               arOrders, iOrders, maOrders,
                               componentsNumberARIMA, componentsNamesARIMA,
                               xregModel, xregModelInitials, xregData, xregNumber, xregNames,
                               xregParametersPersistence);

        # If B is not provided, initialise it
        if(is.null(B)){
            BValues <- initialiser(etsModel, Etype, Ttype, Stype, modelIsTrendy, modelIsSeasonal,
                                   componentsNumberETSNonSeasonal, componentsNumberETSSeasonal, componentsNumberETS,
                                   lags, lagsModel, lagsModelSeasonal, lagsModelARIMA, lagsModelMax,
                                   adamCreated$matVt,
                                   persistenceEstimate, persistenceLevelEstimate, persistenceTrendEstimate,
                                   persistenceSeasonalEstimate, persistenceXregEstimate,
                                   phiEstimate, initialType, initialEstimate,
                                   initialLevelEstimate, initialTrendEstimate, initialSeasonalEstimate,
                                   initialArimaEstimate, initialXregEstimate,
                                   arimaModel, arRequired, maRequired, arEstimate, maEstimate, arOrders, maOrders,
                                   componentsNumberARIMA, componentsNamesARIMA, initialArimaNumber,
                                   xregModel, xregNumber,
                                   xregParametersEstimated, xregParametersPersistence,
                                   otherParameterEstimate);
        }
        # print(BValues$B);

        # Preheat the initial state of ARIMA. Do this only for optimal initials and if B is not provided
        if(arimaModel && initialType=="optimal" && initialArimaEstimate && is.null(B)){
            adamElements <- filler(BValues$B,
                                   etsModel, Etype, Ttype, Stype, modelIsTrendy, modelIsSeasonal,
                                   componentsNumberETS, componentsNumberETSNonSeasonal,
                                   componentsNumberETSSeasonal, componentsNumberARIMA,
                                   lags, lagsModel, lagsModelMax,
                                   adamCreated$matVt, adamCreated$matWt, adamCreated$matF, adamCreated$vecG,
                                   persistenceEstimate, persistenceLevelEstimate, persistenceTrendEstimate,
                                   persistenceSeasonalEstimate, persistenceXregEstimate,
                                   phiEstimate,
                                   initialType, initialEstimate,
                                   initialLevelEstimate, initialTrendEstimate, initialSeasonalEstimate,
                                   initialArimaEstimate, initialXregEstimate,
                                   arimaModel, arEstimate, maEstimate, arOrders, iOrders, maOrders,
                                   arRequired, maRequired, armaParameters,
                                   nonZeroARI, nonZeroMA, adamCreated$arimaPolynomials,
                                   xregModel, xregNumber,
                                   xregParametersMissing, xregParametersIncluded,
                                   xregParametersEstimated, xregParametersPersistence);

            # Write down the initials in the recent profile
            profilesRecentTable[] <- adamElements$matVt[,1:lagsModelMax];

            # Do initial fit to get the state values from the backcasting
            adamFitted <- adamFitterWrap(adamElements$matVt, adamElements$matWt, adamElements$matF, adamElements$vecG,
                                         lagsModelAll, profilesObservedTable, profilesRecentTable,
                                         Etype, Ttype, Stype, componentsNumberETS, componentsNumberETSSeasonal,
                                         componentsNumberARIMA, xregNumber, yInSample, ot, TRUE);

            adamElements$matVt[,1:lagsModelMax] <- adamFitted$matVt[,1:lagsModelMax];
            # Produce new initials
            BValuesNew <- initialiser(etsModel, Etype, Ttype, Stype, modelIsTrendy, modelIsSeasonal,
                                      componentsNumberETSNonSeasonal, componentsNumberETSSeasonal, componentsNumberETS,
                                      lags, lagsModel, lagsModelSeasonal, lagsModelARIMA, lagsModelMax,
                                      adamElements$matVt,
                                      persistenceEstimate, persistenceLevelEstimate, persistenceTrendEstimate,
                                      persistenceSeasonalEstimate, persistenceXregEstimate,
                                      phiEstimate, initialType, initialEstimate,
                                      initialLevelEstimate, initialTrendEstimate, initialSeasonalEstimate,
                                      initialArimaEstimate, initialXregEstimate,
                                      arimaModel, arRequired, maRequired, arEstimate, maEstimate, arOrders, maOrders,
                                      componentsNumberARIMA, componentsNamesARIMA, initialArimaNumber,
                                      xregModel, xregNumber,
                                      xregParametersEstimated, xregParametersPersistence,
                                      otherParameterEstimate);
            B <- BValuesNew$B;
            # Failsafe, just in case if the initial values contain NA / NaN
            if(any(is.na(B))){
                B[is.na(B)] <- BValues$B[is.na(B)];
            }
            if(any(is.nan(B))){
                B[is.nan(B)] <- BValues$B[is.nan(B)];
            }
        }

        # Create the vector of initials for the optimisation
        if(is.null(B)){
            B <- BValues$B
        }
        # if(is.null(lb)){
        #     lb <- BValues$Bl;
        # }
        # if(is.null(ub)){
        #     ub <- BValues$Bu;
        # }

        # Matrices needed for the polynomials calculation -> stationarity / stability checks
        if(arimaModel){
            # AR polynomials
            arPolynomialMatrix <- matrix(0, arOrders %*% lags, arOrders %*% lags);
            if(nrow(arPolynomialMatrix)>1){
                arPolynomialMatrix[2:nrow(arPolynomialMatrix)-1,2:nrow(arPolynomialMatrix)] <- diag(nrow(arPolynomialMatrix)-1);
            }
            # MA polynomials
            maPolynomialMatrix <- matrix(0, maOrders %*% lags, maOrders %*% lags);
            if(nrow(maPolynomialMatrix)>1){
                maPolynomialMatrix[2:nrow(maPolynomialMatrix)-1,2:nrow(maPolynomialMatrix)] <- diag(nrow(maPolynomialMatrix)-1);
            }
        }
        else{
            maPolynomialMatrix <- arPolynomialMatrix <- NULL;
        }

        # If the distribution is default, change it according to the error term
        if(distribution=="default"){
            distributionNew <- switch(loss,
                                      "likelihood"= switch(Etype, "A"= "dnorm", "M"= "dinvgauss"),
                                      "MAEh"=, "MACE"=, "MAE"= "dlaplace",
                                      "HAMh"=, "CHAM"=, "HAM"= "ds",
                                      "MSEh"=, "MSCE"=, "MSE"=, "GPL"=, "dnorm");
        }
        else{
            distributionNew <- distribution;
        }
        # print(B)
        # print(Etype)
        # print(Ttype)
        # print(Stype)
        # stop()

        print_level_hidden <- print_level;
        if(print_level==41){
            print_level[] <- 0;
        }

        maxevalUsed <- maxeval;
        if(is.null(maxeval)){
            maxevalUsed <- length(B) * 40;
            # If this is pure ARIMA, take more time
            if(arimaModel && !etsModel){
                maxevalUsed <- max(1000,maxevalUsed);
            }
            # If it is xregModel, do at least 500 iterations
            else if(xregModel){
                maxevalUsed <- max(500,maxevalUsed);
            }
        }

        # Prepare the denominator needed for the shrinkage of explanatory variables in LASSO / RIDGE
        if(any(loss==c("LASSO","RIDGE")) && xregNumber>0){
            denominator <- apply(matWt, 2, sd);
            denominator[is.infinite(denominator)] <- 1;
        }
        else{
            denominator <- NULL;
        }

        # Parameters are chosen to speed up the optimisation process and have decent accuracy
        res <- suppressWarnings(nloptr(B, CF, lb=lb, ub=ub,
                                       opts=list(algorithm=algorithm, xtol_rel=xtol_rel, xtol_abs=xtol_abs,
                                                 ftol_rel=ftol_rel, ftol_abs=ftol_abs,
                                                 maxeval=maxevalUsed, maxtime=maxtime, print_level=print_level),
                                       etsModel=etsModel, Etype=Etype, Ttype=Ttype, Stype=Stype, modelIsTrendy=modelIsTrendy,
                                       modelIsSeasonal=modelIsSeasonal, yInSample=yInSample,
                                       ot=ot, otLogical=otLogical, occurrenceModel=occurrenceModel, obsInSample=obsInSample,
                                       componentsNumberETS=componentsNumberETS,
                                       componentsNumberETSSeasonal=componentsNumberETSSeasonal,
                                       componentsNumberETSNonSeasonal=componentsNumberETSNonSeasonal,
                                       componentsNumberARIMA=componentsNumberARIMA,
                                       lags=lags, lagsModel=lagsModel, lagsModelAll=lagsModelAll, lagsModelMax=lagsModelMax,
                                       profilesObservedTable=profilesObservedTable, profilesRecentTable=profilesRecentTable,
                                       matVt=adamCreated$matVt, matWt=adamCreated$matWt,
                                       matF=adamCreated$matF, vecG=adamCreated$vecG,
                                       persistenceEstimate=persistenceEstimate, persistenceLevelEstimate=persistenceLevelEstimate,
                                       persistenceTrendEstimate=persistenceTrendEstimate,
                                       persistenceSeasonalEstimate=persistenceSeasonalEstimate,
                                       persistenceXregEstimate=persistenceXregEstimate,
                                       phiEstimate=phiEstimate, initialType=initialType,
                                       initialEstimate=initialEstimate, initialLevelEstimate=initialLevelEstimate,
                                       initialTrendEstimate=initialTrendEstimate, initialSeasonalEstimate=initialSeasonalEstimate,
                                       initialArimaEstimate=initialArimaEstimate, initialXregEstimate=initialXregEstimate,
                                       arimaModel=arimaModel, nonZeroARI=nonZeroARI, nonZeroMA=nonZeroMA,
                                       arimaPolynomials=adamCreated$arimaPolynomials,
                                       arEstimate=arEstimate, maEstimate=maEstimate,
                                       arOrders=arOrders, iOrders=iOrders, maOrders=maOrders,
                                       arRequired=arRequired, maRequired=maRequired, armaParameters=armaParameters,
                                       xregModel=xregModel, xregNumber=xregNumber,
                                       xregParametersMissing=xregParametersMissing,
                                       xregParametersIncluded=xregParametersIncluded,
                                       xregParametersEstimated=xregParametersEstimated,
                                       xregParametersPersistence=xregParametersPersistence,
                                       bounds=bounds, loss=loss, lossFunction=lossFunction, distribution=distributionNew,
                                       horizon=horizon, multisteps=multisteps, denominator=denominator,
                                       other=other, otherParameterEstimate=otherParameterEstimate, lambda=lambda,
                                       arPolynomialMatrix=arPolynomialMatrix, maPolynomialMatrix=maPolynomialMatrix));

        if(is.infinite(res$objective) || res$objective==1e+300){
            # If the optimisation didn't work, give it another try with zero initials for smoothing parameters
            if(etsModel){
                B[1:sum(persistenceLevelEstimate,persistenceTrendEstimate,persistenceSeasonalEstimate)] <- 0;
            }
            if(arimaModel){
                B[sum(persistenceLevelEstimate,persistenceTrendEstimate,persistenceSeasonalEstimate,
                    persistenceXregEstimate*xregNumber)+c(1:sum(arOrders*arEstimate,maOrders*maEstimate))] <- 0.01;
            }
            # print(B)
            res <- suppressWarnings(nloptr(B, CF, lb=lb, ub=ub,
                                           opts=list(algorithm=algorithm, xtol_rel=xtol_rel,
                                                     ftol_rel=ftol_rel, ftol_abs=ftol_abs,
                                                     maxeval=maxevalUsed, maxtime=maxtime, print_level=print_level),
                                           etsModel=etsModel, Etype=Etype, Ttype=Ttype, Stype=Stype, modelIsTrendy=modelIsTrendy,
                                           modelIsSeasonal=modelIsSeasonal, yInSample=yInSample,
                                           ot=ot, otLogical=otLogical, occurrenceModel=occurrenceModel, obsInSample=obsInSample,
                                           componentsNumberETS=componentsNumberETS,
                                           componentsNumberETSSeasonal=componentsNumberETSSeasonal,
                                           componentsNumberETSNonSeasonal=componentsNumberETSNonSeasonal,
                                           componentsNumberARIMA=componentsNumberARIMA,
                                           lags=lags, lagsModel=lagsModel, lagsModelAll=lagsModelAll, lagsModelMax=lagsModelMax,
                                           profilesObservedTable=profilesObservedTable, profilesRecentTable=profilesRecentTable,
                                           matVt=adamCreated$matVt, matWt=adamCreated$matWt,
                                           matF=adamCreated$matF, vecG=adamCreated$vecG,
                                           persistenceEstimate=persistenceEstimate,
                                           persistenceLevelEstimate=persistenceLevelEstimate,
                                           persistenceTrendEstimate=persistenceTrendEstimate,
                                           persistenceSeasonalEstimate=persistenceSeasonalEstimate,
                                           persistenceXregEstimate=persistenceXregEstimate,
                                           phiEstimate=phiEstimate, initialType=initialType,
                                           initialEstimate=initialEstimate, initialLevelEstimate=initialLevelEstimate,
                                           initialTrendEstimate=initialTrendEstimate, initialSeasonalEstimate=initialSeasonalEstimate,
                                           initialArimaEstimate=initialArimaEstimate, initialXregEstimate=initialXregEstimate,
                                           arimaModel=arimaModel, nonZeroARI=nonZeroARI, nonZeroMA=nonZeroMA,
                                           arimaPolynomials=adamCreated$arimaPolynomials,
                                           arEstimate=arEstimate, maEstimate=maEstimate,
                                           arOrders=arOrders, iOrders=iOrders, maOrders=maOrders,
                                           arRequired=arRequired, maRequired=maRequired, armaParameters=armaParameters,
                                           xregModel=xregModel, xregNumber=xregNumber,
                                           xregParametersMissing=xregParametersMissing,
                                           xregParametersIncluded=xregParametersIncluded,
                                           xregParametersEstimated=xregParametersEstimated,
                                           xregParametersPersistence=xregParametersPersistence,
                                           bounds=bounds, loss=loss, lossFunction=lossFunction, distribution=distributionNew,
                                           horizon=horizon, multisteps=multisteps, denominator=denominator,
                                           other=other, otherParameterEstimate=otherParameterEstimate, lambda=lambda,
                                           arPolynomialMatrix=arPolynomialMatrix, maPolynomialMatrix=maPolynomialMatrix));
        }

        if(print_level_hidden>0){
            print(res);
        }

        ##### !!! Check the obtained parameters and the loss value and remove redundant parameters !!! #####
        # Cases to consider:
        # 1. Some smoothing parameters are zero or one;
        # 2. The cost function value is -Inf (due to no variability in the sample);

        # Prepare the values to return
        B[] <- res$solution;
        CFValue <- res$objective;
        # In case of likelihood, we typically have one more parameter to estimate - scale
        nParamEstimated <- length(B) + (loss=="likelihood");
        # Return a proper logLik class
        logLikADAMValue <- structure(logLikADAM(B,
                                                etsModel, Etype, Ttype, Stype, modelIsTrendy, modelIsSeasonal, yInSample,
                                                ot, otLogical, occurrenceModel, pFitted, obsInSample,
                                                componentsNumberETS, componentsNumberETSSeasonal, componentsNumberETSNonSeasonal,
                                                componentsNumberARIMA,
                                                lags, lagsModel, lagsModelAll, lagsModelMax,
                                                profilesObservedTable, profilesRecentTable,
                                                adamCreated$matVt, adamCreated$matWt, adamCreated$matF, adamCreated$vecG,
                                                persistenceEstimate, persistenceLevelEstimate, persistenceTrendEstimate,
                                                persistenceSeasonalEstimate, persistenceXregEstimate,
                                                phiEstimate, initialType, initialEstimate,
                                                initialLevelEstimate, initialTrendEstimate, initialSeasonalEstimate,
                                                initialArimaEstimate, initialXregEstimate,
                                                arimaModel, nonZeroARI, nonZeroMA, arEstimate, maEstimate,
                                                adamCreated$arimaPolynomials,
                                                arOrders, iOrders, maOrders, arRequired, maRequired, armaParameters,
                                                xregModel, xregNumber,
                                                xregParametersMissing, xregParametersIncluded,
                                                xregParametersEstimated, xregParametersPersistence,
                                                bounds, loss, lossFunction, distributionNew, horizon, multisteps,
                                                denominator, other, otherParameterEstimate, lambda,
                                                arPolynomialMatrix, maPolynomialMatrix),
                                     nobs=obsInSample,df=nParamEstimated,class="logLik");
        xregIndex <- 1;
        #### If we do variables selection, do it here, then reestimate the model. ####
        if(regressors=="select"){
            # Fill in the matrices
            adamElements <- filler(B,
                                   etsModel, Etype, Ttype, Stype, modelIsTrendy, modelIsSeasonal,
                                   componentsNumberETS, componentsNumberETSNonSeasonal,
                                   componentsNumberETSSeasonal, componentsNumberARIMA,
                                   lags, lagsModel, lagsModelMax,
                                   adamCreated$matVt, adamCreated$matWt, adamCreated$matF, adamCreated$vecG,
                                   persistenceEstimate, persistenceLevelEstimate, persistenceTrendEstimate,
                                   persistenceSeasonalEstimate, persistenceXregEstimate,
                                   phiEstimate,
                                   initialType, initialEstimate,
                                   initialLevelEstimate, initialTrendEstimate, initialSeasonalEstimate,
                                   initialArimaEstimate, initialXregEstimate,
                                   arimaModel, arEstimate, maEstimate, arOrders, iOrders, maOrders,
                                   arRequired, maRequired, armaParameters,
                                   nonZeroARI, nonZeroMA, adamCreated$arimaPolynomials,
                                   xregModel, xregNumber,
                                   xregParametersMissing, xregParametersIncluded,
                                   xregParametersEstimated, xregParametersPersistence);

            # Write down the initials in the recent profile
            profilesRecentTable[] <- adamElements$matVt[,1:lagsModelMax];

            # Fit the model to the data
            adamFitted <- adamFitterWrap(adamElements$matVt, adamElements$matWt, adamElements$matF, adamElements$vecG,
                                         lagsModelAll, profilesObservedTable, profilesRecentTable,
                                         Etype, Ttype, Stype, componentsNumberETS, componentsNumberETSSeasonal,
                                         componentsNumberARIMA, xregNumber, yInSample, ot, initialType=="backcasting");

            # Extract the errors corrrectly
            errors <- switch(distribution,
                             "dlnorm"=, "dllaplace"=, "dls"=,
                             "dlgnorm"=, "dinvgauss"=switch(Etype,
                                                            "A"=1+adamFitted$errors/adamFitted$yFitted,
                                                            "M"=adamFitted$errors),
                             "dnorm"=, "dlaplace"=, "ds"=, "dgnorm"=, "dlogis"=, "dt"=, "dalaplace"=,adamFitted$errors);
            # Extract the errors and amend them to correspond to the distribution
            errors[] <- errors + switch(Etype,"A"=0,"M"=1);

            if(any(distribution==c("dlnorm","dllaplace","dls","dlgnorm")) && Etype=="A" && any(errors<0)){
                errors[errors<0] <- 1e-100;
            }

            df <- length(B)+1;
            if(any(distribution==c("dalaplace","dgnorm","dlgnorm","dt")) && otherParameterEstimate){
                other <- abs(B[length(B)]);
                df[] <- df - 1;
            }

            # Call the xregSelector providing the original matrix with the data
            xregIndex[] <- switch(Etype,"A"=1,"M"=2);
            xregModelInitials[[xregIndex]] <- xregSelector(errors=errors, xregData=data[,colnames(data)!=responseName,drop=FALSE],
                                                           ic=ic,
                                                           df=df, distribution=distributionNew, occurrence=oesModel,
                                                           other=other);
            xregNumber <- length(xregModelInitials[[xregIndex]]$initialXreg);
            xregNames <- names(xregModelInitials[[xregIndex]]$initialXreg);

            # Fix the names of variables
            xregNames[] <- make.names(xregNames, unique=TRUE);

            # If there are some variables, then do the proper reestimation and return the new values
            if(xregNumber>0){
                xregModel[] <- TRUE;
                initialXregEstimate[] <- initialXregEstimateOriginal;
                persistenceXregEstimate[] <- persistenceXregEstimateOriginal;
                # xregData <- xregDataOriginal[,xregNames,drop=FALSE];

                # Redefine loss for ALM
                lossNew <- switch(loss,
                                  "MSEh"=,"TMSE"=,"GTMSE"=,"MSCE"="MSE",
                                  "MAEh"=,"TMAE"=,"GTMAE"=,"MACE"="MAE",
                                  "HAMh"=,"THAM"=,"GTHAM"=,"CHAM"="HAM",
                                  loss);
                if(lossNew=="custom"){
                    lossNew <- lossFunction;
                }

                # Fix the name of the response variable
                xregModelInitials[[xregIndex]]$formula[[2]] <- as.name(responseName);
                formula <- xregModelInitials[[xregIndex]]$formula;
                xregModelInitials[[which(c(1,2)!=xregIndex)]]$formula <- formula;

                # Estimate alm again in order to get proper initials
                almModel <- do.call(alm,list(formula=formula,
                                             data=data[1:obsInSample,,drop=FALSE],
                                             distribution=distributionNew, loss=lossNew, occurrence=oesModel));
                xregModelInitials[[xregIndex]]$initialXreg <- coef(almModel)[-1];

                #### Fix xreg vectors based on the selected stuff ####
                xregNames <- colnames(almModel$data[,-1]);
                # Robustify the names of variables
                colnames(data) <- make.names(colnames(data),unique=TRUE);
                # The names of the original variables
                xregNamesOriginal <- all.vars(formula)[-1];
                # Levels for the factors
                xregFactorsLevels <- lapply(data,levels);
                xregFactorsLevels[[responseName]] <- NULL;
                # Expand the variables. We cannot use alm, because it is based on obsInSample
                xregData <- model.frame(formula,data=as.data.frame(data));
                # Binary, flagging factors in the data
                xregFactors <- (attr(terms(xregData),"dataClasses")=="factor")[-1];
                # Expanded stuff with all levels for factors
                if(any(xregFactors)){
                    xregModelMatrix <- model.matrix(xregData,xregData,
                                                    contrasts.arg=lapply(xregData[attr(terms(xregData),"dataClasses")=="factor"],
                                                                         contrasts, contrasts=FALSE));
                    xregNamesModified <- colnames(xregModelMatrix)[-1];
                }
                else{
                    xregModelMatrix <- model.matrix(xregData,data=xregData);
                    xregNamesModified <- xregNames;
                }
                xregData <- as.matrix(xregModelMatrix);
                # Remove intercept
                interceptIsPresent <- FALSE;
                if(any(colnames(xregData)=="(Intercept)")){
                    interceptIsPresent[] <- TRUE;
                    xregData <- xregData[,-1,drop=FALSE];
                }
                xregNumber <- ncol(xregData);

                # The indices of the original parameters
                xregParametersMissing <- setNames(vector("numeric",xregNumber),xregNamesModified);
                # # The indices of the original parameters
                xregParametersIncluded <- setNames(vector("numeric",xregNumber),xregNamesModified);
                # The vector, marking the same values of smoothing parameters
                if(interceptIsPresent){
                    xregParametersPersistence <- setNames(attr(xregModelMatrix,"assign")[-1],xregNamesModified);
                }
                else{
                    xregParametersPersistence <- setNames(attr(xregModelMatrix,"assign"),xregNamesModified);
                }

                # If there are factors not in the alm data, create additional initials
                if(any(!(xregNamesModified %in% xregNames))){
                    xregAbsent <- !(xregNamesModified %in% xregNames);
                    xregParametersNew <- setNames(rep(NA,xregNumber),xregNamesModified);
                    xregParametersNew[!xregAbsent] <- xregModelInitials[[xregIndex]]$initialXreg;
                    # Go through new names and find, where they came from. Then get the missing parameters
                    for(i in which(xregAbsent)){
                        # Find the name of the original variable
                        # Use only the last value... hoping that the names like x and x1 are not used.
                        xregNameFound <- tail(names(sapply(xregNamesOriginal,grepl,xregNamesModified[i])),1);
                        # Get the indices of all k-1 levels
                        xregParametersIncluded[xregNames[xregNames %in% paste0(xregNameFound,
                                                                               xregFactorsLevels[[xregNameFound]])]] <- i;
                        # Get the index of the absent one
                        xregParametersMissing[i] <- i;
                        # Fill in the absent one
                        xregParametersNew[i] <- -sum(xregParametersNew[xregNamesModified[xregParametersIncluded==i]],
                                                     na.rm=TRUE);
                    }
                    # Write down the new parameters
                    xregModelInitials[[xregIndex]]$initialXreg <- xregParametersNew;
                    xregNames <- xregNamesModified;
                }
                # The vector of parameters that should be estimated (numeric + original levels of factors)
                xregParametersEstimated <- xregParametersIncluded
                xregParametersEstimated[xregParametersEstimated!=0] <- 1;
                xregParametersEstimated[xregParametersMissing==0 & xregParametersIncluded==0] <- 1;

                return(estimator(etsModel, Etype, Ttype, Stype, lags, lagsModelSeasonal, lagsModelARIMA,
                                 obsStates, obsInSample,
                                 yInSample, persistence, persistenceEstimate,
                                 persistenceLevel, persistenceLevelEstimate,
                                 persistenceTrend, persistenceTrendEstimate,
                                 persistenceSeasonal, persistenceSeasonalEstimate,
                                 persistenceXreg, persistenceXregEstimate, persistenceXregProvided,
                                 phi, phiEstimate,
                                 initialType, initialLevel, initialTrend, initialSeasonal,
                                 initialArima, initialEstimate,
                                 initialLevelEstimate, initialTrendEstimate, initialSeasonalEstimate,
                                 initialArimaEstimate, initialXregEstimate, initialXregProvided,
                                 arimaModel, arRequired, iRequired, maRequired, armaParameters,
                                 componentsNumberARIMA, componentsNamesARIMA,
                                 formula, xregModel, xregModelInitials, xregData, xregNumber, xregNames, regressors="use",
                                 xregParametersMissing, xregParametersIncluded,
                                 xregParametersEstimated, xregParametersPersistence,
                                 ot, otLogical, occurrenceModel, pFitted,
                                 bounds, loss, lossFunction, distribution,
                                 horizon, multisteps, other, otherParameterEstimate, lambda));
            }
        }

        return(list(B=B, CFValue=CFValue, nParamEstimated=nParamEstimated, logLikADAMValue=logLikADAMValue,
                    xregModel=xregModel, xregData=xregData, xregNumber=xregNumber,
                    xregNames=xregNames, xregModelInitials=xregModelInitials, formula=formula,
                    initialXregEstimate=initialXregEstimate, persistenceXregEstimate=persistenceXregEstimate,
                    xregParametersMissing=xregParametersMissing,xregParametersIncluded=xregParametersIncluded,
                    xregParametersEstimated=xregParametersEstimated,xregParametersPersistence=xregParametersPersistence,
                    arimaPolynomials=adamCreated$arimaPolynomials));
    }


    #### The function creates a pool of models and selects the best of them ####
    selector <- function(model, modelsPool, allowMultiplicative,
                         etsModel, Etype, Ttype, Stype, damped, lags,
                         lagsModelSeasonal, lagsModelARIMA,
                         obsStates, obsInSample,
                         yInSample, persistence, persistenceEstimate,
                         persistenceLevel, persistenceLevelEstimate,
                         persistenceTrend, persistenceTrendEstimate,
                         persistenceSeasonal, persistenceSeasonalEstimate,
                         persistenceXreg, persistenceXregEstimate, persistenceXregProvided,
                         phi, phiEstimate,
                         initialType, initialLevel, initialTrend, initialSeasonal,
                         initialArima, initialEstimate,
                         initialLevelEstimate, initialTrendEstimate, initialSeasonalEstimate,
                         initialArimaEstimate, initialXregEstimate, initialXregProvided,
                         arimaModel, arRequired, iRequired, maRequired, armaParameters,
                         componentsNumberARIMA, componentsNamesARIMA,
                         xregModel, xregModelInitials, xregData, xregNumber, xregNames, regressors,
                         xregParametersMissing, xregParametersIncluded,
                         xregParametersEstimated, xregParametersPersistence,
                         ot, otLogical, occurrenceModel, pFitted, ICFunction,
                         bounds, loss, lossFunction, distribution,
                         horizon, multisteps, other, otherParameterEstimate, lambda){

        # Check if the pool was provided. In case of "no", form the big and the small ones
        if(is.null(modelsPool)){
            # The variable saying that the pool was not provided.
            if(!silent){
                cat("Forming the pool of models based on... ");
            }

            # Define the whole pool of errors
            if(!allowMultiplicative){
                poolErrors <- c("A");
                poolTrends <- c("N","A","Ad");
                poolSeasonals <- c("N","A");
            }
            else{
                poolErrors <- c("A","M");
                poolTrends <- c("N","A","Ad","M","Md");
                poolSeasonals <- c("N","A","M");
            }

            # Some preparation variables
            # If Etype is not Z, then check on additive errors
            if(Etype!="Z"){
                poolErrors <- poolErrorsSmall <- Etype;
            }
            else{
                poolErrorsSmall <- "A";
            }

            # If Ttype is not Z, then create a pool with specified type
            if(Ttype!="Z"){
                if(Ttype=="X"){
                    poolTrendsSmall <- c("N","A");
                    poolTrends <- c("N","A","Ad");
                    checkTrend <- TRUE;
                }
                else if(Ttype=="Y"){
                    poolTrendsSmall <- c("N","M");
                    poolTrends <- c("N","M","Md");
                    checkTrend <- TRUE;
                }
                else{
                    if(damped){
                        poolTrends <- poolTrendsSmall <- paste0(Ttype,"d");
                    }
                    else{
                        poolTrends <- poolTrendsSmall <- Ttype;
                    }
                    checkTrend <- FALSE;
                }
            }
            else{
                poolTrendsSmall <- c("N","A");
                checkTrend <- TRUE;
            }

            # If Stype is not Z, then crete specific pools
            if(Stype!="Z"){
                if(Stype=="X"){
                    poolSeasonals <- poolSeasonalsSmall <- c("N","A");
                    checkSeasonal <- TRUE;
                }
                else if(Stype=="Y"){
                    poolSeasonalsSmall <- c("N","M");
                    poolSeasonals <- c("N","M");
                    checkSeasonal <- TRUE;
                }
                else{
                    poolSeasonalsSmall <- Stype;
                    poolSeasonals <- Stype;
                    checkSeasonal <- FALSE;
                }
            }
            else{
                poolSeasonalsSmall <- c("N","A","M");
                checkSeasonal <- TRUE;
            }

            # If ZZZ, then the vector is: "ANN" "ANA" "ANM" "AAN" "AAA" "AAM"
            # Otherwise id depends on the provided restrictions
            poolSmall <- paste0(rep(poolErrorsSmall,length(poolTrendsSmall)*length(poolSeasonalsSmall)),
                                rep(poolTrendsSmall,each=length(poolSeasonalsSmall)),
                                rep(poolSeasonalsSmall,length(poolTrendsSmall)));
            # Align error and seasonality, if the error was not forced to be additive
            if(any(substr(poolSmall,3,3)=="M") && all(Etype!=c("A","X"))){
                multiplicativeSeason <- (substr(poolSmall,3,3)=="M");
                poolSmall[multiplicativeSeason] <- paste0("M",substr(poolSmall[multiplicativeSeason],2,3));
            }
            modelsTested <- NULL;
            modelCurrent <- NA;

            # Counter + checks for the components
            j <- 1;
            i <- 0;
            check <- TRUE;
            besti <- bestj <- 1;
            results <- vector("list",length(poolSmall));

            #### Branch and bound is here ####
            while(check){
                i <- i + 1;
                modelCurrent[] <- poolSmall[j];
                if(!silent){
                    cat(modelCurrent,"\b, ");
                }
                Etype[] <- substring(modelCurrent,1,1);
                Ttype[] <- substring(modelCurrent,2,2);
                if(nchar(modelCurrent)==4){
                    phi[] <- 0.95;
                    phiEstimate[] <- TRUE;
                    Stype[] <- substring(modelCurrent,4,4);
                }
                else{
                    phi[] <- 1;
                    phiEstimate[] <- FALSE;
                    Stype[] <- substring(modelCurrent,3,3);
                }

                results[[i]] <- estimator(etsModel, Etype, Ttype, Stype, lags, lagsModelSeasonal, lagsModelARIMA,
                                          obsStates, obsInSample,
                                          yInSample, persistence, persistenceEstimate,
                                          persistenceLevel, persistenceLevelEstimate,
                                          persistenceTrend, persistenceTrendEstimate,
                                          persistenceSeasonal, persistenceSeasonalEstimate,
                                          persistenceXreg, persistenceXregEstimate, persistenceXregProvided,
                                          phi, phiEstimate,
                                          initialType, initialLevel, initialTrend, initialSeasonal,
                                          initialArima, initialEstimate,
                                          initialLevelEstimate, initialTrendEstimate, initialSeasonalEstimate,
                                          initialArimaEstimate, initialXregEstimate, initialXregProvided,
                                          arimaModel, arRequired, iRequired, maRequired, armaParameters,
                                          componentsNumberARIMA, componentsNamesARIMA,
                                          formula, xregModel, xregModelInitials, xregData, xregNumber, xregNames, regressors,
                                          xregParametersMissing, xregParametersIncluded,
                                          xregParametersEstimated, xregParametersPersistence,
                                          ot, otLogical, occurrenceModel, pFitted,
                                          bounds, loss, lossFunction, distribution,
                                          horizon, multisteps, other, otherParameterEstimate, lambda);
                results[[i]]$IC <- ICFunction(results[[i]]$logLikADAMValue);
                results[[i]]$Etype <- Etype;
                results[[i]]$Ttype <- Ttype;
                results[[i]]$Stype <- Stype;
                results[[i]]$phiEstimate <- phiEstimate;
                if(phiEstimate){
                    results[[i]]$phi <- results[[i]]$B[names(results[[i]]$B)=="phi"];
                }
                else{
                    results[[i]]$phi <- 1;
                }
                results[[i]]$model <- modelCurrent;

                modelsTested <- c(modelsTested,modelCurrent);

                if(j>1){
                    # If the first is better than the second, then choose first
                    if(results[[besti]]$IC <= results[[i]]$IC){
                        # If Ttype is the same, then we check seasonality
                        if(substring(modelCurrent,2,2)==substring(poolSmall[bestj],2,2)){
                            poolSeasonals <- results[[besti]]$Stype;
                            checkSeasonal <- FALSE;
                            j <- which(poolSmall!=poolSmall[bestj] &
                                           substring(poolSmall,nchar(poolSmall),nchar(poolSmall))==poolSeasonals);
                        }
                        # Otherwise we checked trend
                        else{
                            poolTrends <- results[[bestj]]$Ttype;
                            checkTrend[] <- FALSE;
                        }
                    }
                    else{
                        # If the trend is the same
                        if(substring(modelCurrent,2,2) == substring(poolSmall[besti],2,2)){
                            poolSeasonals <- poolSeasonals[poolSeasonals!=results[[besti]]$Stype];
                            if(length(poolSeasonals)>1){
                                # Select another seasonal model, that is not from the previous iteration and not the current one
                                bestj[] <- j;
                                besti[] <- i;
                                # j[] <- 3;
                                j <- 3;
                            }
                            else{
                                bestj[] <- j;
                                besti[] <- i;
                                # Move to checking the trend
                                j <- which(substring(poolSmall,nchar(poolSmall),nchar(poolSmall))==poolSeasonals &
                                               substring(poolSmall,2,2)!=substring(modelCurrent,2,2));
                                checkSeasonal[] <- FALSE;
                            }
                        }
                        else{
                            poolTrends <- poolTrends[poolTrends!=results[[bestj]]$Ttype];
                            besti[] <- i;
                            bestj[] <- j;
                            checkTrend[] <- FALSE;
                        }
                    }

                    if(all(!c(checkTrend,checkSeasonal))){
                        check[] <- FALSE;
                    }
                }
                else{
                    j <- 2;
                }

                if(j>length(poolSmall)){
                    check[] <- FALSE;
                }
            }

            # Prepare a bigger pool based on the small one
            modelsPool <- unique(c(modelsTested,
                                   paste0(rep(poolErrors,each=length(poolTrends)*length(poolSeasonals)),
                                          poolTrends,
                                          rep(poolSeasonals,each=length(poolTrends)))));
            j <- length(modelsTested);
        }
        else{
            j <- 0;
            results <- vector("list",length(modelsPool));
        }
        modelsNumber <- length(modelsPool);

        #### Run the full pool of models ####
        if(!silent){
            cat("Estimation progress:    ");
        }
        # Start loop of models
        while(j < modelsNumber){
            j <- j + 1;
            if(!silent){
                if(j==1){
                    cat("\b");
                }
                cat(paste0(rep("\b",nchar(round((j-1)/modelsNumber,2)*100)+1),collapse=""));
                cat(round(j/modelsNumber,2)*100,"\b%");
            }

            modelCurrent <- modelsPool[j];
            # print(modelCurrent)
            Etype <- substring(modelCurrent,1,1);
            Ttype <- substring(modelCurrent,2,2);
            if(nchar(modelCurrent)==4){
                phi[] <- 0.95;
                Stype <- substring(modelCurrent,4,4);
                phiEstimate <- TRUE;
            }
            else{
                phi[] <- 1;
                Stype <- substring(modelCurrent,3,3);
                phiEstimate <- FALSE;
            }

            results[[j]] <- estimator(etsModel, Etype, Ttype, Stype, lags, lagsModelSeasonal, lagsModelARIMA,
                                      obsStates, obsInSample,
                                      yInSample, persistence, persistenceEstimate,
                                      persistenceLevel, persistenceLevelEstimate,
                                      persistenceTrend, persistenceTrendEstimate,
                                      persistenceSeasonal, persistenceSeasonalEstimate,
                                      persistenceXreg, persistenceXregEstimate, persistenceXregProvided,
                                      phi, phiEstimate,
                                      initialType, initialLevel, initialTrend, initialSeasonal,
                                      initialArima, initialEstimate,
                                      initialLevelEstimate, initialTrendEstimate, initialSeasonalEstimate,
                                      initialArimaEstimate, initialXregEstimate, initialXregProvided,
                                      arimaModel, arRequired, iRequired, maRequired, armaParameters,
                                      componentsNumberARIMA, componentsNamesARIMA,
                                      formula, xregModel, xregModelInitials, xregData, xregNumber, xregNames, regressors,
                                      xregParametersMissing, xregParametersIncluded,
                                      xregParametersEstimated, xregParametersPersistence,
                                      ot, otLogical, occurrenceModel, pFitted,
                                      bounds, loss, lossFunction, distribution,
                                      horizon, multisteps, other, otherParameterEstimate, lambda);
            results[[j]]$IC <- ICFunction(results[[j]]$logLikADAMValue);
            results[[j]]$Etype <- Etype;
            results[[j]]$Ttype <- Ttype;
            results[[j]]$Stype <- Stype;
            results[[j]]$phiEstimate <- phiEstimate;
            if(phiEstimate){
                results[[j]]$phi <- results[[j]]$B[names(results[[j]]$B)=="phi"];
            }
            else{
                results[[j]]$phi <- 1;
            }
            results[[j]]$model <- modelCurrent;
        }

        if(!silent){
            cat("... Done! \n");
        }

        # Extract ICs and find the best
        icSelection <- vector("numeric",modelsNumber);
        for(i in 1:modelsNumber){
            icSelection[i] <- results[[i]]$IC;
        }
        names(icSelection) <- modelsPool;

        icSelection[is.nan(icSelection)] <- 1E100;

        return(list(results=results,icSelection=icSelection));
    }

    ##### Function uses residuals in order to determine the needed xreg #####
    xregSelector <- function(errors, xregData, ic, df, distribution, occurrence, other){
        alpha <- beta <- nu <- NULL;
        if(distribution=="dalaplace"){
            alpha <- other;
        }
        else if(any(distribution==c("dgnorm","dlgnorm"))){
            beta <- other;
        }
        else if(distribution=="dt"){
            nu <- other;
        }
        stepwiseModel <- suppressWarnings(stepwise(cbind(as.data.frame(errors),xregData[1:obsInSample,,drop=FALSE]),
                                                   ic=ic, df=df, distribution=distribution, occurrence=occurrence, silent=TRUE,
                                                   alpha=alpha, beta=beta, nu=nu));
        return(list(initialXreg=coef(stepwiseModel)[-1],other=stepwiseModel$other,formula=formula(stepwiseModel)));
    }

    ##### Function prepares all the matrices and vectors for return #####
    preparator <- function(B, etsModel, Etype, Ttype, Stype,
                           lagsModel, lagsModelMax, lagsModelAll,
                           componentsNumberETS, componentsNumberETSSeasonal,
                           xregNumber, distribution, loss,
                           persistenceEstimate, persistenceLevelEstimate, persistenceTrendEstimate,
                           persistenceSeasonalEstimate, persistenceXregEstimate,
                           phiEstimate, otherParameterEstimate,
                           initialType, initialEstimate,
                           initialLevelEstimate, initialTrendEstimate, initialSeasonalEstimate,
                           initialArimaEstimate, initialXregEstimate,
                           matVt, matWt, matF, vecG,
                           occurrenceModel, ot, oesModel,
                           parametersNumber, CFValue,
                           arimaModel, arRequired, maRequired,
                           arEstimate, maEstimate, arOrders, iOrders, maOrders,
                           nonZeroARI, nonZeroMA,
                           arimaPolynomials, armaParameters){

        if(modelDo!="use"){
            # Fill in the matrices
            adamElements <- filler(B,
                                   etsModel, Etype, Ttype, Stype, modelIsTrendy, modelIsSeasonal,
                                   componentsNumberETS, componentsNumberETSNonSeasonal,
                                   componentsNumberETSSeasonal, componentsNumberARIMA,
                                   lags, lagsModel, lagsModelMax,
                                   matVt, matWt, matF, vecG,
                                   persistenceEstimate, persistenceLevelEstimate, persistenceTrendEstimate,
                                   persistenceSeasonalEstimate, persistenceXregEstimate,
                                   phiEstimate,
                                   initialType, initialEstimate,
                                   initialLevelEstimate, initialTrendEstimate, initialSeasonalEstimate,
                                   initialArimaEstimate, initialXregEstimate,
                                   arimaModel, arEstimate, maEstimate, arOrders, iOrders, maOrders,
                                   arRequired, maRequired, armaParameters,
                                   nonZeroARI, nonZeroMA, arimaPolynomials,
                                   xregModel, xregNumber,
                                   xregParametersMissing, xregParametersIncluded,
                                   xregParametersEstimated, xregParametersPersistence);
            list2env(adamElements, environment());
        }

        # Write down phi
        if(phiEstimate){
            phi[] <- B[names(B)=="phi"];
        }

        # Write down the initials in the recent profile
        profilesRecentTable[] <- matVt[,1:lagsModelMax];

        # Fit the model to the data
        adamFitted <- adamFitterWrap(matVt, matWt, matF, vecG,
                                     lagsModelAll, profilesObservedTable, profilesRecentTable,
                                     Etype, Ttype, Stype, componentsNumberETS, componentsNumberETSSeasonal,
                                     componentsNumberARIMA, xregNumber, yInSample, ot, initialType=="backcasting");

        matVt[] <- adamFitted$matVt;

        # Write down the recent profile for future use
        profilesRecentTable <- adamFitted$profile;

        if(any(yClasses=="ts")){
            yFitted <- ts(rep(NA,obsInSample), start=yStart, frequency=yFrequency);
            errors <- ts(rep(NA,obsInSample), start=yStart, frequency=yFrequency);
        }
        else{
            yFitted <- zoo(rep(NA,obsInSample), order.by=yInSampleIndex);
            errors <- zoo(rep(NA,obsInSample), order.by=yInSampleIndex);
        }

        errors[] <- adamFitted$errors;
        yFitted[] <- adamFitted$yFitted;
        # Check what was returned in the end
        if(any(is.nan(yFitted)) || any(is.na(yFitted))){
            warning("Something went wrong in the estimation of the model ETS(",
                    Etype,Ttype,ifelse(damped,"d",""),Stype,") and NaNs were produced. ",
                    "If this is a mixed model, consider using the pure ones instead.",
                    call.=FALSE);
        }
        if(occurrenceModel){
            yFitted[] <- yFitted * pFitted;
        }

        # Produce forecasts if the horizon is non-zero
        if(horizon>0){
            if(any(yClasses=="ts")){
                yForecast <- ts(rep(NA, horizon), start=yForecastStart, frequency=yFrequency);
            }
            else{
                yForecast <- zoo(rep(NA, horizon), order.by=yForecastIndex);
            }

            yForecast[] <- adamForecasterWrap(tail(matWt,horizon), matF,
                                              lagsModelAll,
                                              profilesObservedTable[,obsInSample+c(1:horizon),drop=FALSE],profilesRecentTable,
                                              Etype, Ttype, Stype,
                                              componentsNumberETS, componentsNumberETSSeasonal,
                                              componentsNumberARIMA, xregNumber,
                                              horizon);
            #### Make safety checks
            # If there are NaN values
            if(any(is.nan(yForecast))){
                yForecast[is.nan(yForecast)] <- 0;
            }

            # Amend forecasts, multiplying by probability
            if(occurrenceModel && !occurrenceModelProvided){
                yForecast[] <- yForecast * c(forecast(oesModel, h=h)$mean);
            }
            else if(occurrenceModel && occurrenceModelProvided){
                yForecast[] <- yForecast * pForecast;
            }
        }
        else{
            if(any(yClasses=="ts")){
                yForecast <- ts(NA, start=yForecastStart, frequency=yFrequency);
            }
            else{
                yForecast <- zoo(rep(NA, horizon), order.by=yForecastIndex);
            }
        }

        # If the distribution is default, change it according to the error term
        if(distribution=="default"){
            distribution[] <- switch(loss,
                                     "likelihood"= switch(Etype, "A"= "dnorm", "M"= "dinvgauss"),
                                     "MAEh"=, "MACE"=, "MAE"= "dlaplace",
                                     "HAMh"=, "CHAM"=, "HAM"= "ds",
                                     "MSEh"=, "MSCE"=, "MSE"=, "GPL"=, "dnorm");
        }

        #### Initial values to return ####
        initialValue <- vector("list", etsModel*(1+modelIsTrendy+modelIsSeasonal)+arimaModel+xregModel);
        initialValueETS <- vector("list", etsModel*length(lagsModel));
        initialValueNames <- vector("character", etsModel*(1+modelIsTrendy+modelIsSeasonal)+arimaModel+xregModel);
        # The vector that defines what was estimated in the model
        initialEstimated <- vector("logical", etsModel*(1+modelIsTrendy+modelIsSeasonal*componentsNumberETSSeasonal)+
                                       arimaModel+xregModel);

        # Write down the initials of ETS
        j <- 0;
        if(etsModel){
            # Write down level, trend and seasonal
            for(i in 1:length(lagsModel)){
                initialValueETS[[i]] <- tail(matVt[i,1:lagsModelMax],lagsModel[i]);
            }
            j[] <- j+1;
            # Write down level in the final list
            initialEstimated[j] <- initialLevelEstimate;
            initialValue[[j]] <- initialValueETS[[j]];
            initialValueNames[j] <- c("level");
            names(initialEstimated)[j] <- initialValueNames[j];
            if(modelIsTrendy){
                j[] <- 2;
                initialEstimated[j] <- initialTrendEstimate;
                # Write down trend in the final list
                initialValue[[j]] <- initialValueETS[[j]];
                # Remove the trend from ETS list
                initialValueETS[[j]] <- NULL;
                initialValueNames[j] <- c("trend");
                names(initialEstimated)[j] <- initialValueNames[j];
            }
            # Write down the initial seasonals
            if(modelIsSeasonal){
                initialEstimated[j+c(1:componentsNumberETSSeasonal)] <- initialSeasonalEstimate;
                # Remove the level from ETS list
                initialValueETS[[1]] <- NULL;
                j[] <- j+1;
                if(length(initialSeasonalEstimate)>1){
                    initialValue[[j]] <- initialValueETS;
                    initialValueNames[[j]] <- "seasonal";
                    names(initialEstimated)[j+0:(componentsNumberETSSeasonal-1)] <-
                        paste0(initialValueNames[j],c(1:componentsNumberETSSeasonal));
                }
                else{
                    initialValue[[j]] <- initialValueETS[[1]];
                    initialValueNames[[j]] <- "seasonal";
                    names(initialEstimated)[j] <- initialValueNames[j];
                }
            }
        }

        # Write down the ARIMA initials
        if(arimaModel){
            j[] <- j+1;
            initialEstimated[j] <- initialArimaEstimate;
            if(initialArimaEstimate && initialType=="optimal"){
                initialValue[[j]] <- B[substr(names(B),1,10)=="ARIMAState"];
            }
            else if(initialArimaEstimate && initialType=="backcasting"){
                initialValue[[j]] <- head(matVt[componentsNumberETS+componentsNumberARIMA,],initialArimaNumber);
            }
            else{
                initialValue[[j]] <- initialArima;
            }
            initialValueNames[j] <- "arima";
            names(initialEstimated)[j] <- initialValueNames[j];
        }
        # Write down the xreg initials
        if(xregModel){
            j[] <- j+1;
            initialEstimated[j] <- initialXregEstimate;
            initialValue[[j]] <- matVt[componentsNumberETS+componentsNumberARIMA+1:xregNumber,lagsModelMax];
            initialValueNames[j] <- "xreg";
            names(initialEstimated)[j] <- initialValueNames[j];
        }
        names(initialValue) <- initialValueNames;

        #### Persistence to return ####
        persistence <- as.vector(vecG);
        names(persistence) <- rownames(vecG);

        # Remove xreg persistence from the returned vector
        if(xregModel && regressors!="adapt"){
            persistence <- persistence[substr(names(persistence),1,5)!="delta"];
            # We've selected the variables, so there's nothing to select anymore
            regressors <- "use";
        }
        else if(!xregModel){
            regressors <- NULL;
        }


        if(arimaModel){
            armaParametersList <- vector("list",arRequired+maRequired);
            j[] <- 1;
            if(arRequired && arEstimate){
                # Avoid damping parameter phi
                armaParametersList[[j]] <- B[nchar(names(B))>3 & substr(names(B),1,3)=="phi"];
                names(armaParametersList)[j] <- "ar";
                j[] <- j+1;
            }
            # If this was provided
            else if(arRequired && !arEstimate){
                # Avoid damping parameter phi
                armaParametersList[[j]] <- armaParameters[substr(names(armaParameters),1,3)=="phi"];
                names(armaParametersList)[j] <- "ar";
                j[] <- j+1;
            }
            if(maRequired && maEstimate){
                armaParametersList[[j]] <- B[substr(names(B),1,5)=="theta"];
                names(armaParametersList)[j] <- "ma";
            }
            else if(maRequired && !maEstimate){
                armaParametersList[[j]] <- armaParameters[substr(names(armaParameters),1,5)=="theta"];
                names(armaParametersList)[j] <- "ma";
            }
        }
        else{
            armaParametersList <- NULL;
        }

        if(any(distribution==c("dalaplace","dgnorm","dlgnorm","dt")) && otherParameterEstimate){
            other <- abs(tail(B,1));
        }
        # which() is needed in order to overcome weird behaviour of zoo
        scale <- scaler(distribution, Etype, errors[which(otLogical)], yFitted[which(otLogical)], obsInSample, other);

        # Prepare the list of distribution parameters to return
        otherReturned <- vector("list",1);
        # Write down parameters for distribution. It is always positive, so take abs
        if(otherParameterEstimate){
            otherReturned[[1]] <- abs(tail(B,1));
        }
        else{
            otherReturned[[1]] <- other;
        }
        # Give names to the other values
        if(distribution=="dalaplace"){
            names(otherReturned) <- "alpha";
        }
        else if(any(distribution==c("dgnorm","dlgnorm"))){
            names(otherReturned) <- "beta";
        }
        else if(any(distribution==c("dt"))){
            names(otherReturned) <- "nu";
        }
        # LASSO / RIDGE lambda
        if(any(loss==c("LASSO","RIDGE"))){
            otherReturned$lambda <- lambda;
        }
        # Return ARIMA polynomials and indices for persistence and transition
        if(arimaModel){
            otherReturned$polynomial <- arimaPolynomials;
            otherReturned$ARIMAIndices <- list(nonZeroARI=nonZeroARI,nonZeroMA=nonZeroMA);
            otherReturned$arPolynomialMatrix <- matrix(0, arOrders %*% lags, arOrders %*% lags);
            if(nrow(otherReturned$arPolynomialMatrix)>1){
                otherReturned$arPolynomialMatrix[2:nrow(otherReturned$arPolynomialMatrix)-1,
                                                 2:nrow(otherReturned$arPolynomialMatrix)] <-
                    diag(nrow(otherReturned$arPolynomialMatrix)-1);
                if(arRequired){
                    otherReturned$arPolynomialMatrix[,1] <- -arimaPolynomials$arPolynomial[-1];
                }
            }
            otherReturned$armaParameters <- armaParameters;
        }

        # Amend the class of state matrix
        if(any(yClasses=="ts")){
            matVt <- ts(t(matVt), start=(time(y)[1]-deltat(y)*lagsModelMax), frequency=yFrequency);
        }
        else{
            yStatesIndex <- yInSampleIndex[1] - lagsModelMax*diff(tail(yInSampleIndex,2)) +
                c(1:lagsModelMax-1)*diff(tail(yInSampleIndex,2));
            yStatesIndex <- c(yStatesIndex, yInSampleIndex);
            matVt <- zoo(t(matVt), order.by=yStatesIndex);
        }

        parametersNumber[2,4] <- sum(parametersNumber[2,1:3]);

        return(list(model=NA, timeElapsed=NA,
                    data=cbind(NA,xregData), holdout=NULL, fitted=yFitted, residuals=errors,
                    forecast=yForecast, states=matVt, profile=profilesRecentTable,
                    persistence=persistence, phi=phi, transition=matF,
                    measurement=matWt, initial=initialValue, initialType=initialType,
                    initialEstimated=initialEstimated, orders=orders, arma=armaParametersList,
                    nParam=parametersNumber, occurrence=oesModel,
                    formula=formula, regressors=regressors,
                    loss=loss, lossValue=CFValue, logLik=logLikADAMValue, distribution=distribution,
                    scale=scale, other=otherReturned, B=B, lags=lags, lagsAll=lagsModelAll, FI=FI));
    }

    #### Deal with occurrence model ####
    if(occurrenceModel && !occurrenceModelProvided){
        modelForOES <- model;
        if(model=="NNN"){
            modelForOES[] <- "MNN";
        }
        oesModel <- suppressWarnings(oes(ot, model=modelForOES, occurrence=occurrence, ic=ic, h=horizon,
                                         holdout=FALSE, bounds="usual", xreg=xregData, regressors=regressors, silent=TRUE));
        pFitted[] <- fitted(oesModel);
        parametersNumber[1,3] <- nparam(oesModel);
        # print(oesModel)
        # This should not happen, but just in case...
        if(oesModel$occurrence=="n"){
            occurrence <- "n";
            otLogical <- rep(TRUE,obsInSample);
            occurrenceModel <- FALSE;
            ot <- matrix(otLogical*1,ncol=1);
            obsNonzero <- sum(ot);
            obsZero <- obsInSample - obsNonzero;
            Etype[] <- switch(Etype,
                              "M"="A",
                              "Y"=,
                              "Z"="X",
                              Etype);
            Ttype[] <- switch(Ttype,
                              "M"="A",
                              "Y"=,
                              "Z"="X",
                              Ttype);
            Stype[] <- switch(Stype,
                              "M"="A",
                              "Y"=,
                              "Z"="X",
                              Stype);
        }
    }
    else if(occurrenceModel && occurrenceModelProvided){
        parametersNumber[2,3] <- nparam(oesModel);
    }

    xregDataOriginal <- xregData;
    ##### Prepare stuff for the variables selection if regressors="select" #####
    if(regressors=="select"){
        # First, record the original parameters
        xregExistOriginal <- xregModel;
        initialXregsProvidedOriginal <- initialXregProvided;
        initialXregEstimateOriginal <- initialXregEstimate;
        persistenceXregOriginal <- persistenceXreg;
        persistenceXregProvidedOriginal <- persistenceXregProvided;
        persistenceXregEstimateOriginal <- persistenceXregEstimate;
        xregModelOriginal <- xregModelInitials;
        xregNumberOriginal <- xregNumber;
        xregNamesOriginal <- xregNames;

        # Set the parameters to zero and do simple ETS
        xregModel[] <- FALSE;
        initialXregProvided <- FALSE;
        initialXregEstimate[] <- FALSE;
        persistenceXreg <- 0;
        persistenceXregProvided <- FALSE;
        persistenceXregEstimate[] <- FALSE;
        xregData <- NULL;
        xregNumber[] <- 0;
        xregNames <- NULL;
    }

    ##### Estimate the specified model #####
    if(modelDo=="estimate"){
        # Estimate the parameters of the demand sizes model
        adamEstimated <- estimator(etsModel, Etype, Ttype, Stype, lags, lagsModelSeasonal, lagsModelARIMA,
                                   obsStates, obsInSample,
                                   yInSample, persistence, persistenceEstimate,
                                   persistenceLevel, persistenceLevelEstimate,
                                   persistenceTrend, persistenceTrendEstimate,
                                   persistenceSeasonal, persistenceSeasonalEstimate,
                                   persistenceXreg, persistenceXregEstimate, persistenceXregProvided,
                                   phi, phiEstimate,
                                   initialType, initialLevel, initialTrend, initialSeasonal,
                                   initialArima, initialEstimate,
                                   initialLevelEstimate, initialTrendEstimate, initialSeasonalEstimate,
                                   initialArimaEstimate, initialXregEstimate, initialXregProvided,
                                   arimaModel, arRequired, iRequired, maRequired, armaParameters,
                                   componentsNumberARIMA, componentsNamesARIMA,
                                   formula, xregModel, xregModelInitials, xregData, xregNumber, xregNames, regressors,
                                   xregParametersMissing, xregParametersIncluded,
                                   xregParametersEstimated, xregParametersPersistence,
                                   ot, otLogical, occurrenceModel, pFitted,
                                   bounds, loss, lossFunction, distribution,
                                   horizon, multisteps, other, otherParameterEstimate, lambda);
        list2env(adamEstimated, environment());

        #### This part is needed in order for the filler to do its job later on
        # Create the basic variables based on the estimated model
        adamArchitect <- architector(etsModel, Etype, Ttype, Stype, lags, lagsModelSeasonal,
                                     xregNumber, obsInSample, initialType,
                                     arimaModel, lagsModelARIMA, xregModel,
                                     profilesRecentTable, profilesRecentProvided);
        list2env(adamArchitect, environment());

        # Create the matrices for the specific ETS model
        adamCreated <- creator(etsModel, Etype, Ttype, Stype, modelIsTrendy, modelIsSeasonal,
                               lags, lagsModel, lagsModelARIMA, lagsModelAll, lagsModelMax,
                               profilesRecentTable, profilesRecentProvided,
                               obsStates, obsInSample, obsAll, componentsNumberETS, componentsNumberETSSeasonal,
                               componentsNamesETS, otLogical, yInSample,
                               persistence, persistenceEstimate,
                               persistenceLevel, persistenceLevelEstimate, persistenceTrend, persistenceTrendEstimate,
                               persistenceSeasonal, persistenceSeasonalEstimate,
                               persistenceXreg, persistenceXregEstimate, persistenceXregProvided,
                               phi,
                               initialType, initialEstimate,
                               initialLevel, initialLevelEstimate, initialTrend, initialTrendEstimate,
                               initialSeasonal, initialSeasonalEstimate,
                               initialArima, initialArimaEstimate, initialArimaNumber,
                               initialXregEstimate, initialXregProvided,
                               arimaModel, arRequired, iRequired, maRequired, armaParameters,
                               arOrders, iOrders, maOrders,
                               componentsNumberARIMA, componentsNamesARIMA,
                               xregModel, xregModelInitials, xregData, xregNumber, xregNames,
                               xregParametersPersistence);
        list2env(adamCreated, environment());

        icSelection <- ICFunction(adamEstimated$logLikADAMValue);

        ####!!! If the occurrence is auto, then compare this with the model with no occurrence !!!####

        parametersNumber[1,1] <- nParamEstimated;
        if(xregModel){
            parametersNumber[1,2] <- sum(xregParametersEstimated)*initialXregEstimate +
                max(xregParametersPersistence)*persistenceXregEstimate;
            parametersNumber[1,1] <- parametersNumber[1,1] - parametersNumber[1,2]
        }
        parametersNumber[1,4] <- sum(parametersNumber[1,1:3]);
        parametersNumber[2,4] <- sum(parametersNumber[2,1:3]);
    }
    #### Selection of the best model ####
    else if(modelDo=="select"){
        adamSelected <-  selector(model, modelsPool, allowMultiplicative,
                                  etsModel, Etype, Ttype, Stype, damped, lags,
                                  lagsModelSeasonal, lagsModelARIMA,
                                  obsStates, obsInSample,
                                  yInSample, persistence, persistenceEstimate,
                                  persistenceLevel, persistenceLevelEstimate,
                                  persistenceTrend, persistenceTrendEstimate,
                                  persistenceSeasonal, persistenceSeasonalEstimate,
                                  persistenceXreg, persistenceXregEstimate, persistenceXregProvided,
                                  phi, phiEstimate,
                                  initialType, initialLevel, initialTrend, initialSeasonal,
                                  initialArima, initialEstimate,
                                  initialLevelEstimate, initialTrendEstimate, initialSeasonalEstimate,
                                  initialArimaEstimate, initialXregEstimate, initialXregProvided,
                                  arimaModel, arRequired, iRequired, maRequired, armaParameters,
                                  componentsNumberARIMA, componentsNamesARIMA,
                                  xregModel, xregModelInitials, xregData, xregNumber, xregNames, regressors,
                                  xregParametersMissing, xregParametersIncluded,
                                  xregParametersEstimated, xregParametersPersistence,
                                  ot, otLogical, occurrenceModel, pFitted, ICFunction,
                                  bounds, loss, lossFunction, distribution,
                                  horizon, multisteps, other, otherParameterEstimate, lambda);

        icSelection <- adamSelected$icSelection;
        # Take the parameters of the best model
        list2env(adamSelected$results[[which.min(icSelection)[1]]], environment());

        #### This part is needed in order for the filler to do its job later on
        # Create the basic variables based on the estimated model
        adamArchitect <- architector(etsModel, Etype, Ttype, Stype, lags, lagsModelSeasonal,
                                     xregNumber, obsInSample, initialType,
                                     arimaModel, lagsModelARIMA, xregModel,
                                     profilesRecentTable, profilesRecentProvided);
        list2env(adamArchitect, environment());

        # Create the matrices for the specific ETS model
        adamCreated <- creator(etsModel, Etype, Ttype, Stype, modelIsTrendy, modelIsSeasonal,
                               lags, lagsModel, lagsModelARIMA, lagsModelAll, lagsModelMax,
                               profilesRecentTable, profilesRecentProvided,
                               obsStates, obsInSample, obsAll, componentsNumberETS, componentsNumberETSSeasonal,
                               componentsNamesETS, otLogical, yInSample,
                               persistence, persistenceEstimate,
                               persistenceLevel, persistenceLevelEstimate, persistenceTrend, persistenceTrendEstimate,
                               persistenceSeasonal, persistenceSeasonalEstimate,
                               persistenceXreg, persistenceXregEstimate, persistenceXregProvided,
                               phi,
                               initialType, initialEstimate,
                               initialLevel, initialLevelEstimate, initialTrend, initialTrendEstimate,
                               initialSeasonal, initialSeasonalEstimate,
                               initialArima, initialArimaEstimate, initialArimaNumber,
                               initialXregEstimate, initialXregProvided,
                               arimaModel, arRequired, iRequired, maRequired, armaParameters,
                               arOrders, iOrders, maOrders,
                               componentsNumberARIMA, componentsNamesARIMA,
                               xregModel, xregModelInitials, xregData, xregNumber, xregNames,
                               xregParametersPersistence);
        list2env(adamCreated, environment());

        parametersNumber[1,1] <- nParamEstimated;
        if(xregModel){
            parametersNumber[1,2] <- xregNumber*initialXregEstimate + xregNumber*persistenceXregEstimate;
        }
        parametersNumber[1,4] <- sum(parametersNumber[1,1:3]);
        parametersNumber[2,4] <- sum(parametersNumber[2,1:3]);
    }
    #### Combination of models ####
    else if(modelDo=="combine"){
        modelOriginal <- model;
        # If the pool is not provided, then create one
        if(is.null(modelsPool)){
            # Define the whole pool of errors
            if(!allowMultiplicative){
                poolErrors <- c("A");
                poolTrends <- c("N","A","Ad");
                poolSeasonals <- c("N","A");
            }
            else{
                poolErrors <- c("A","M");
                poolTrends <- c("N","A","Ad","M","Md");
                poolSeasonals <- c("N","A","M");
            }

            # Some preparation variables
            # If Etype is not Z, then check on additive errors
            if(Etype!="Z"){
                poolErrors <- switch(Etype,
                                     "N"="N",
                                     "A"=,
                                     "X"="A",
                                     "M"=,
                                     "Y"="M");
            }

            # If Ttype is not Z, then create a pool with specified type
            if(Ttype!="Z"){
                poolTrends <- switch(Ttype,
                                     "N"="N",
                                     "A"=ifelse(damped,"Ad","A"),
                                     "M"=ifelse(damped,"Md","M"),
                                     "X"=c("N","A","Ad"),
                                     "Y"=c("N","M","Md"));
            }

            # If Stype is not Z, then crete specific pools
            if(Stype!="Z"){
                poolSeasonals <- switch(Stype,
                                        "N"="N",
                                        "A"="A",
                                        "X"=c("N","A"),
                                        "M"="M",
                                        "Y"=c("N","M"));
            }

            modelsPool <- paste0(rep(poolErrors,length(poolTrends)*length(poolSeasonals)),
                                 rep(poolTrends,each=length(poolSeasonals)),
                                 rep(poolSeasonals,length(poolTrends)));
        }

        adamSelected <-  selector(model, modelsPool, allowMultiplicative,
                                  etsModel, Etype, Ttype, Stype, damped, lags,
                                  lagsModelSeasonal, lagsModelARIMA,
                                  obsStates, obsInSample,
                                  yInSample, persistence, persistenceEstimate,
                                  persistenceLevel, persistenceLevelEstimate,
                                  persistenceTrend, persistenceTrendEstimate,
                                  persistenceSeasonal, persistenceSeasonalEstimate,
                                  persistenceXreg, persistenceXregEstimate, persistenceXregProvided,
                                  phi, phiEstimate,
                                  initialType, initialLevel, initialTrend, initialSeasonal,
                                  initialArima, initialEstimate,
                                  initialLevelEstimate, initialTrendEstimate, initialSeasonalEstimate,
                                  initialArimaEstimate, initialXregEstimate, initialXregProvided,
                                  arimaModel, arRequired, iRequired, maRequired, armaParameters,
                                  componentsNumberARIMA, componentsNamesARIMA,
                                  xregModel, xregModelInitials, xregData, xregNumber, xregNames, regressors,
                                  xregParametersMissing, xregParametersIncluded,
                                  xregParametersEstimated, xregParametersPersistence,
                                  ot, otLogical, occurrenceModel, pFitted, ICFunction,
                                  bounds, loss, lossFunction, distribution,
                                  horizon, multisteps, other, otherParameterEstimate, lambda);

        icSelection <- adamSelected$icSelection;

        icBest <- min(icSelection);
        adamSelected$icWeights  <- (exp(-0.5*(icSelection-icBest)) /
                                        sum(exp(-0.5*(icSelection-icBest))));

        # This is a failsafe mechanism, just to make sure that the ridiculous models don't impact forecasts
        adamSelected$icWeights[adamSelected$icWeights<1e-5] <- 0
        adamSelected$icWeights <- adamSelected$icWeights/sum(adamSelected$icWeights);

        # adamArchitect <- vector("list",10)
        for(i in 1:length(adamSelected$results)){
            # Take the parameters of the best model
            list2env(adamSelected$results[[i]], environment());

            #### This part is needed in order for the filler to do its job later on
            # Create the basic variables based on the estimated model
            adamArchitect <- architector(etsModel, Etype, Ttype, Stype, lags, lagsModelSeasonal,
                                         xregNumber, obsInSample, initialType,
                                         arimaModel, lagsModelARIMA, xregModel,
                                         profilesRecentTable, profilesRecentProvided);
            list2env(adamArchitect, environment());

            adamSelected$results[[i]]$modelIsTrendy <- adamArchitect$modelIsTrendy;
            adamSelected$results[[i]]$modelIsSeasonal <- adamArchitect$modelIsSeasonal;
            adamSelected$results[[i]]$lagsModel <- adamArchitect$lagsModel;
            adamSelected$results[[i]]$lagsModelAll <- adamArchitect$lagsModelAll;
            adamSelected$results[[i]]$lagsModelMax <- adamArchitect$lagsModelMax;
            adamSelected$results[[i]]$profilesRecentTable <- adamArchitect$profilesRecentTable;
            adamSelected$results[[i]]$profilesObservedTable <- adamArchitect$profilesObservedTable;
            adamSelected$results[[i]]$componentsNumberETS <- adamArchitect$componentsNumberETS;
            adamSelected$results[[i]]$componentsNumberETSSeasonal <- adamArchitect$componentsNumberETSSeasonal;
            adamSelected$results[[i]]$componentsNumberETSNonSeasonal <- adamArchitect$componentsNumberETSNonSeasonal;
            adamSelected$results[[i]]$componentsNamesETS <- adamArchitect$componentsNamesETS;

            # Create the matrices for the specific ETS model
            adamCreated <- creator(etsModel, Etype, Ttype, Stype, modelIsTrendy, modelIsSeasonal,
                                   lags, lagsModel, lagsModelARIMA, lagsModelAll, lagsModelMax,
                                   profilesRecentTable, profilesRecentProvided,
                                   obsStates, obsInSample, obsAll, componentsNumberETS, componentsNumberETSSeasonal,
                                   componentsNamesETS, otLogical, yInSample,
                                   persistence, persistenceEstimate,
                                   persistenceLevel, persistenceLevelEstimate, persistenceTrend, persistenceTrendEstimate,
                                   persistenceSeasonal, persistenceSeasonalEstimate,
                                   persistenceXreg, persistenceXregEstimate, persistenceXregProvided,
                                   phi,
                                   initialType, initialEstimate,
                                   initialLevel, initialLevelEstimate, initialTrend, initialTrendEstimate,
                                   initialSeasonal, initialSeasonalEstimate,
                                   initialArima, initialArimaEstimate, initialArimaNumber,
                                   initialXregEstimate, initialXregProvided,
                                   arimaModel, arRequired, iRequired, maRequired, armaParameters,
                                   arOrders, iOrders, maOrders,
                                   componentsNumberARIMA, componentsNamesARIMA,
                                   xregModel, xregModelInitials, xregData, xregNumber, xregNames,
                                   xregParametersPersistence);

            adamSelected$results[[i]]$matVt <- adamCreated$matVt;
            adamSelected$results[[i]]$matWt <- adamCreated$matWt;
            adamSelected$results[[i]]$matF <- adamCreated$matF;
            adamSelected$results[[i]]$vecG <- adamCreated$vecG;
            adamSelected$results[[i]]$arimaPolynomials <- adamCreated$arimaPolynomials;

            parametersNumber[1,1] <- adamSelected$results[[i]]$nParamEstimated;
            if(xregModel){
                parametersNumber[1,2] <- xregNumber*initialXregEstimate + xregNumber*persistenceXregEstimate;
            }
            parametersNumber[1,4] <- sum(parametersNumber[1,1:3]);
            parametersNumber[2,4] <- sum(parametersNumber[2,1:3]);

            adamSelected$results[[i]]$parametersNumber <- parametersNumber;
        }
    }
    #### Use the provided model ####
    else if(modelDo=="use"){
        # If the distribution is default, change it according to the error term
        if(distribution=="default"){
            distributionNew <- switch(loss,
                                      "likelihood"= switch(Etype, "A"= "dnorm", "M"= "dinvgauss"),
                                      "MAEh"=, "MACE"=, "MAE"= "dlaplace",
                                      "HAMh"=, "CHAM"=, "HAM"= "ds",
                                      "MSEh"=, "MSCE"=, "MSE"=, "GPL"=, "dnorm");
        }
        else{
            distributionNew <- distribution;
        }

        # Create the basic variables
        adamArchitect <- architector(etsModel, Etype, Ttype, Stype, lags, lagsModelSeasonal,
                                     xregNumber, obsInSample, initialType,
                                     arimaModel, lagsModelARIMA, xregModel,
                                     profilesRecentTable, profilesRecentProvided);
        list2env(adamArchitect, environment());

        # Create the matrices for the specific ETS model
        adamCreated <- creator(etsModel, Etype, Ttype, Stype, modelIsTrendy, modelIsSeasonal,
                               lags, lagsModel, lagsModelARIMA, lagsModelAll, lagsModelMax,
                               profilesRecentTable, profilesRecentProvided,
                               obsStates, obsInSample, obsAll, componentsNumberETS, componentsNumberETSSeasonal,
                               componentsNamesETS, otLogical, yInSample,
                               persistence, persistenceEstimate,
                               persistenceLevel, persistenceLevelEstimate, persistenceTrend, persistenceTrendEstimate,
                               persistenceSeasonal, persistenceSeasonalEstimate,
                               persistenceXreg, persistenceXregEstimate, persistenceXregProvided,
                               phi,
                               initialType, initialEstimate,
                               initialLevel, initialLevelEstimate, initialTrend, initialTrendEstimate,
                               initialSeasonal, initialSeasonalEstimate,
                               initialArima, initialArimaEstimate, initialArimaNumber,
                               initialXregEstimate, initialXregProvided,
                               arimaModel, arRequired, iRequired, maRequired, armaParameters,
                               arOrders, iOrders, maOrders,
                               componentsNumberARIMA, componentsNamesARIMA,
                               xregModel, xregModelInitials, xregData, xregNumber, xregNames,
                               xregParametersPersistence);
        list2env(adamCreated, environment());

        # Prepare the denominator needed for the shrinkage of explanatory variables in LASSO / RIDGE
        if(xregNumber>0 && any(loss==c("LASSO","RIDGE"))){
            denominator <- apply(matWt, 2, sd);
            denominator[is.infinite(denominator)] <- 1;
        }
        else{
            denominator <- NULL;
        }

        CFValue <- CF(B=0, etsModel=etsModel, Etype=Etype, Ttype=Ttype, Stype=Stype, modelIsTrendy=modelIsTrendy,
                      modelIsSeasonal=modelIsSeasonal, yInSample=yInSample,
                      ot=ot, otLogical=otLogical, occurrenceModel=occurrenceModel, obsInSample=obsInSample,
                      componentsNumberETS=componentsNumberETS, componentsNumberETSSeasonal=componentsNumberETSSeasonal,
                      componentsNumberETSNonSeasonal=componentsNumberETSNonSeasonal,
                      componentsNumberARIMA=componentsNumberARIMA,
                      lags=lags, lagsModel=lagsModel, lagsModelAll=lagsModelAll, lagsModelMax=lagsModelMax,
                      profilesObservedTable=profilesObservedTable, profilesRecentTable=profilesRecentTable,
                      matVt=matVt, matWt=matWt, matF=matF, vecG=vecG,
                      persistenceEstimate=persistenceEstimate,
                      persistenceLevelEstimate=persistenceLevelEstimate,
                      persistenceTrendEstimate=persistenceTrendEstimate,
                      persistenceSeasonalEstimate=persistenceSeasonalEstimate,
                      persistenceXregEstimate=persistenceXregEstimate,
                      phiEstimate=phiEstimate, initialType=initialType,
                      initialEstimate=initialEstimate, initialLevelEstimate=initialLevelEstimate,
                      initialTrendEstimate=initialTrendEstimate, initialSeasonalEstimate=initialSeasonalEstimate,
                      initialArimaEstimate=initialArimaEstimate, initialXregEstimate=initialXregEstimate,
                      arimaModel=arimaModel, nonZeroARI=nonZeroARI, nonZeroMA=nonZeroMA,
                      arimaPolynomials=arimaPolynomials,
                      arEstimate=arEstimate, maEstimate=maEstimate,
                      arOrders=arOrders, iOrders=iOrders, maOrders=maOrders,
                      arRequired=arRequired, maRequired=maRequired, armaParameters=armaParameters,
                      xregModel=xregModel, xregNumber=xregNumber,
                      xregParametersMissing=xregParametersMissing,
                      xregParametersIncluded=xregParametersIncluded,
                      xregParametersEstimated=xregParametersEstimated,
                      xregParametersPersistence=xregParametersPersistence,
                      bounds=bounds, loss=loss, lossFunction=lossFunction, distribution=distributionNew,
                      horizon=horizon, multisteps=multisteps, denominator=denominator,
                      other=other, otherParameterEstimate=otherParameterEstimate, lambda=lambda,
                      arPolynomialMatrix=NULL, maPolynomialMatrix=NULL);

        parametersNumber[1,1] <- parametersNumber[1,4] <- 1;
        logLikADAMValue <- structure(logLikADAM(B=0,
                                                etsModel, Etype, Ttype, Stype, modelIsTrendy, modelIsSeasonal, yInSample,
                                                ot, otLogical, occurrenceModel, pFitted, obsInSample,
                                                componentsNumberETS, componentsNumberETSSeasonal, componentsNumberETSNonSeasonal,
                                                componentsNumberARIMA,
                                                lags, lagsModel, lagsModelAll, lagsModelMax,
                                                profilesObservedTable, profilesRecentTable,
                                                matVt, matWt, matF, vecG,
                                                persistenceEstimate, persistenceLevelEstimate, persistenceTrendEstimate,
                                                persistenceSeasonalEstimate, persistenceXregEstimate,
                                                phiEstimate, initialType, initialEstimate,
                                                initialLevelEstimate, initialTrendEstimate, initialSeasonalEstimate,
                                                initialArimaEstimate, initialXregEstimate,
                                                arimaModel, nonZeroARI, nonZeroMA, arEstimate, maEstimate, arimaPolynomials,
                                                arOrders, iOrders, maOrders, arRequired, maRequired, armaParameters,
                                                xregModel, xregNumber,
                                                xregParametersMissing, xregParametersIncluded,
                                                xregParametersEstimated, xregParametersPersistence,
                                                bounds, loss, lossFunction, distributionNew, horizon,
                                                multisteps, denominator, other, otherParameterEstimate, lambda,
                                                arPolynomialMatrix=NULL, maPolynomialMatrix=NULL)
                                     ,nobs=obsInSample,df=parametersNumber[1,4],class="logLik")

        icSelection <- ICFunction(logLikADAMValue);
        # If Fisher Information is required, do that analytically
        if(FI){
            # If B is not provided, then use the standard thing
            if(is.null(B)){
                BValues <- initialiser(etsModel, Etype, Ttype, Stype, modelIsTrendy, modelIsSeasonal,
                                       componentsNumberETSNonSeasonal, componentsNumberETSSeasonal, componentsNumberETS,
                                       lags, lagsModel, lagsModelSeasonal, lagsModelARIMA, lagsModelMax,
                                       matVt,
                                       TRUE, TRUE, modelIsTrendy, rep(modelIsSeasonal,componentsNumberETSSeasonal), FALSE,
                                       damped, "optimal", TRUE,
                                       TRUE, TRUE, rep(modelIsSeasonal,componentsNumberETSSeasonal),
                                       arimaModel, xregModel,
                                       arimaModel, arRequired, maRequired, arRequired, maRequired, arOrders, maOrders,
                                       componentsNumberARIMA, componentsNamesARIMA, initialArimaNumber,
                                       xregModel, xregNumber,
                                       xregParametersEstimated, xregParametersPersistence,
                                       FALSE);
                # Create the vector of initials for the optimisation
                B <- BValues$B;
            }

            # Reset persistence, just to make sure that there are no duplicates
            vecG[] <- 0;

            initialTypeFI <- switch(initialType,"backcasting"="provided",initialType);
            initialEstimateFI <- FALSE;
            # Define parameters just for FI calculation
            if(initialTypeFI=="provided"){
                initialLevelEstimateFI <- any(names(B)=="level");
                initialTrendEstimateFI <- any(names(B)=="trend");
                if(any(substr(names(B),1,8)=="seasonal")){
                    initialSeasonalEstimateFI <- vector("logical", componentsNumberETSSeasonal);
                    seasonalNames <- names(B)[substr(names(B),1,8)=="seasonal"];
                    # If there is only one seasonality
                    if(any(substr(seasonalNames,1,9)=="seasonal_")){
                        initialSeasonalEstimateFI[] <- TRUE;
                    }
                    # If there are several
                    else{
                        initialSeasonalEstimateFI[unique(as.numeric(substr(seasonalNames,9,9)))] <- TRUE;
                    }
                }
                else{
                    initialSeasonalEstimateFI <- FALSE;
                }

                if(arimaModel){
                    initialArimaEstimateFI <- any(substr(names(B),1,10)=="ARIMAState");
                }
                else{
                    initialArimaEstimateFI <- FALSE;
                }

                if(xregModel){
                    initialXregEstimateFI <- any(colnames(xregData) %in% names(B));
                }
                else{
                    initialXregEstimateFI <- FALSE;
                }

                initialTypeFI <- "optimal";
                initialEstimateFI <- any(c(initialLevelEstimateFI,initialTrendEstimateFI,initialSeasonalEstimateFI,
                                           initialArimaEstimateFI, initialXregEstimateFI));
            }

            # If smoothing parmaeters were estimated, then alpha should be in the list
            persistenceLevelEstimateFI <- any(names(B)=="alpha");
            persistenceTrendEstimateFI <- any(names(B)=="beta");
            if(any(substr(names(B),1,5)=="gamma")){
                gammas <- (substr(names(B),1,5)=="gamma");
                if(sum(gammas)==1){
                    persistenceSeasonalEstimateFI <- TRUE;
                }
                else{
                    persistenceSeasonalEstimateFI <- vector("logical",componentsNumberETSSeasonal);
                    persistenceSeasonalEstimateFI[as.numeric(substr(names(B),6,6)[gammas])] <- TRUE;
                }
            }
            else{
                persistenceSeasonalEstimateFI <- FALSE;
            }
            persistenceXregEstimateFI <- any(substr(names(B),1,5)=="delta");
            persistenceEstimateFI <- any(c(persistenceLevelEstimateFI,persistenceTrendEstimateFI,
                                           persistenceSeasonalEstimateFI,persistenceXregEstimateFI));
            phiEstimateFI <- any(names(B)=="phi");
            otherParameterEstimateFI <- any(names(B)=="other");

            if(arimaModel){
                maEstimateFI <- maRequired;
                arEstimateFI <- arRequired;
                maPolynomialMatrix <- arPolynomialMatrix <- NULL;
            }

            # This is needed in order to avoid the 1e+300 in the CF
            boundsFI <- "none";

            FI <- -hessian(logLikADAM, B, etsModel=etsModel, Etype=Etype, Ttype=Ttype, Stype=Stype, modelIsTrendy=modelIsTrendy,
                           modelIsSeasonal=modelIsSeasonal, yInSample=yInSample,
                           ot=ot, otLogical=otLogical, occurrenceModel=occurrenceModel, pFitted=pFitted, obsInSample=obsInSample,
                           componentsNumberETS=componentsNumberETS, componentsNumberETSSeasonal=componentsNumberETSSeasonal,
                           componentsNumberETSNonSeasonal=componentsNumberETSNonSeasonal,
                           componentsNumberARIMA=componentsNumberARIMA,
                           lags=lags, lagsModel=lagsModel, lagsModelAll=lagsModelAll, lagsModelMax=lagsModelMax,
                           profilesObservedTable=profilesObservedTable, profilesRecentTable=profilesRecentTable,
                           matVt=matVt, matWt=matWt, matF=matF, vecG=vecG,
                           persistenceEstimate=persistenceEstimateFI, persistenceLevelEstimate=persistenceLevelEstimateFI,
                           persistenceTrendEstimate=persistenceTrendEstimateFI,
                           persistenceSeasonalEstimate=persistenceSeasonalEstimateFI,
                           persistenceXregEstimate=persistenceXregEstimateFI,
                           phiEstimate=phiEstimateFI, initialType=initialTypeFI,
                           initialEstimate=initialEstimateFI, initialLevelEstimate=initialLevelEstimateFI,
                           initialTrendEstimate=initialTrendEstimateFI, initialSeasonalEstimate=initialSeasonalEstimateFI,
                           initialArimaEstimate=initialArimaEstimateFI, initialXregEstimate=initialXregEstimateFI,
                           arimaModel=arimaModel, nonZeroARI=nonZeroARI, nonZeroMA=nonZeroMA,
                           arEstimate=arEstimateFI, maEstimate=maEstimateFI, arimaPolynomials=arimaPolynomials,
                           arOrders=arOrders, iOrders=iOrders, maOrders=maOrders,
                           arRequired=arRequired, maRequired=maRequired, armaParameters=armaParameters,
                           xregModel=xregModel, xregNumber=xregNumber,
                           xregParametersMissing=xregParametersMissing,
                           xregParametersIncluded=xregParametersIncluded,
                           xregParametersEstimated=xregParametersEstimated,
                           xregParametersPersistence=xregParametersPersistence,
                           bounds=boundsFI, loss=loss, lossFunction=lossFunction, distribution=distribution,
                           horizon=horizon, multisteps=multisteps, denominator=denominator,
                           other=other, otherParameterEstimate=otherParameterEstimateFI, lambda=lambda,
                           arPolynomialMatrix=arPolynomialMatrix, maPolynomialMatrix=maPolynomialMatrix,
                           hessianCalculation=FALSE,h=stepSize);

            colnames(FI) <- names(B);
            rownames(FI) <- names(B);
        }
        else{
            FI <- NULL;
        }
    }

    # Transform everything into appropriate classes
    if(any(yClasses=="ts")){
        yInSample <- ts(yInSample,start=yStart, frequency=yFrequency);
        if(holdout){
            yHoldout <- ts(as.matrix(yHoldout), start=yForecastStart, frequency=yFrequency);
        }
    }
    else{
        yInSample <- zoo(yInSample, order.by=yInSampleIndex);
        if(holdout){
            yHoldout <- zoo(as.matrix(yHoldout), order.by=yForecastIndex);
        }
    }

    #### Prepare the return if we didn't combine anything ####
    if(modelDo!="combine"){
        modelReturned <- preparator(B, etsModel, Etype, Ttype, Stype,
                                    lagsModel, lagsModelMax, lagsModelAll,
                                    componentsNumberETS, componentsNumberETSSeasonal,
                                    xregNumber, distribution, loss,
                                    persistenceEstimate, persistenceLevelEstimate, persistenceTrendEstimate,
                                    persistenceSeasonalEstimate, persistenceXregEstimate,
                                    phiEstimate, otherParameterEstimate,
                                    initialType, initialEstimate,
                                    initialLevelEstimate, initialTrendEstimate, initialSeasonalEstimate,
                                    initialArimaEstimate, initialXregEstimate,
                                    matVt, matWt, matF, vecG,
                                    occurrenceModel, ot, oesModel,
                                    parametersNumber, CFValue,
                                    arimaModel, arRequired, maRequired,
                                    arEstimate, maEstimate, arOrders, iOrders, maOrders,
                                    nonZeroARI, nonZeroMA,
                                    arimaPolynomials, armaParameters);

        # Prepare the name of the model
        modelName <- "";
        if(etsModel){
            if(model!="NNN"){
                modelName[] <- "ETS";
                if(xregModel){
                    modelName[] <- paste0(modelName,"X");
                }
                modelName[] <- paste0(modelName,"(",model,")");
                if(componentsNumberETSSeasonal>1){
                    modelName[] <- paste0(modelName,"[",paste0(lags[lags!=1], collapse=", "),"]");
                }
            }
            else{
                modelName[] <- "Constant level";
            }
        }
        if(arimaModel){
            if(etsModel){
                modelName[] <- paste0(modelName,"+");
            }
            # Either the lags are non-seasonal, or there are no orders for seasonal lags
            if(all(lags==1) || (all(arOrders[lags>1]==0) && all(iOrders[lags>1]==0) && all(maOrders[lags>1]==0))){
                modelName[] <- paste0(modelName,"ARIMA");
                if(!etsModel && xregModel){
                    modelName[] <- paste0(modelName,"X");
                }
                modelName[] <- paste0(modelName,"(",arOrders[1],",",iOrders[1],",",maOrders[1],")");
            }
            else{
                modelName[] <- paste0(modelName,"SARIMA");
                if(!etsModel && xregModel){
                    modelName[] <- paste0(modelName,"X");
                }
                for(i in 1:length(arOrders)){
                    if(all(arOrders[i]==0) && all(iOrders[i]==0) && all(maOrders[i]==0)){
                        next;
                    }
                    modelName[] <- paste0(modelName,"(",arOrders[i],",");
                    modelName[] <- paste0(modelName,iOrders[i],",");
                    modelName[] <- paste0(modelName,maOrders[i],")[",lags[i],"]");
                }
            }
        }
        if(regressors=="adapt"){
            modelName[] <- paste0(modelName,"{D}");
        }
        if(!etsModel && !arimaModel){
            if(regressors=="adapt"){
                modelName[] <- paste0("Dynamic regression");
            }
            else{
                modelName[] <- paste0("Regression");
            }
        }
        if(all(occurrence!=c("n","none"))){
            modelName[] <- paste0("i",modelName);
        }

        modelReturned$model <- modelName;
        modelReturned$timeElapsed <- Sys.time()-startTime;
        if(!is.null(xregData)){
            # Remove redundant columns from the data
            modelReturned$data <- data[1:obsInSample,,drop=FALSE]
            if(holdout){
                modelReturned$holdout <- data[obsInSample+c(1:h),,drop=FALSE];
            }
            # Fix the ts class, which is destroyed during subsetting
            if(all(yClasses!="zoo")){
                if(is.data.frame(data)){
                    modelReturned$data[,responseName] <- ts(modelReturned$data[,responseName],
                                                            start=yStart, frequency=yFrequency);
                    if(holdout){
                        modelReturned$holdout[,responseName] <- ts(modelReturned$holdout[,responseName],
                                                                   start=yForecastStart, frequency=yFrequency);
                    }
                }
                else{
                    modelReturned$data <- ts(modelReturned$data, start=yStart, frequency=yFrequency);
                    if(holdout){
                        modelReturned$holdout <- ts(modelReturned$holdout, start=yForecastStart, frequency=yFrequency);
                    }
                }
            }
        }
        else{
            modelReturned$data <- yInSample;
            modelReturned$holdout <- yHoldout;
        }
        if(any(yNAValues)){
            modelReturned$data[yNAValues[1:obsInSample],responseName] <- NA;
            if(length(yNAValues)==obsAll){
                modelReturned$holdout[yNAValues[-c(1:obsInSample)],responseName] <- NA;
            }
            modelReturned$residuals[yNAValues[1:obsInSample]] <- NA;
        }

        class(modelReturned) <- c("adam","smooth");
    }
    #### Return the combined model ####
    else{
        modelReturned <- list(models=vector("list",length(adamSelected$results)));
        yFittedCombined <- rep(0,obsInSample);
        if(h>0){
            yForecastCombined <- rep(0,h);
        }
        else{
            yForecastCombined <- NA;
        }
        parametersNumberOverall <- parametersNumber;

        for(i in 1:length(adamSelected$results)){
            list2env(adamSelected$results[[i]], environment());
            modelReturned$models[[i]] <- preparator(B, etsModel, Etype, Ttype, Stype,
                                                    lagsModel, lagsModelMax, lagsModelAll,
                                                    componentsNumberETS, componentsNumberETSSeasonal,
                                                    xregNumber, distribution, loss,
                                                    persistenceEstimate, persistenceLevelEstimate, persistenceTrendEstimate,
                                                    persistenceSeasonalEstimate, persistenceXregEstimate,
                                                    phiEstimate, otherParameterEstimate,
                                                    initialType, initialEstimate,
                                                    initialLevelEstimate, initialTrendEstimate, initialSeasonalEstimate,
                                                    initialArimaEstimate, initialXregEstimate,
                                                    matVt, matWt, matF, vecG,
                                                    occurrenceModel, ot, oesModel,
                                                    parametersNumber, CFValue,
                                                    arimaModel, arRequired, maRequired,
                                                    arEstimate, maEstimate, arOrders, iOrders, maOrders,
                                                    nonZeroARI, nonZeroMA,
                                                    arimaPolynomials, armaParameters);
            modelReturned$models[[i]]$fitted[is.na(modelReturned$models[[i]]$fitted)] <- 0;
            yFittedCombined[] <- yFittedCombined + modelReturned$models[[i]]$fitted * adamSelected$icWeights[i];
            if(h>0){
                modelReturned$models[[i]]$forecast[is.na(modelReturned$models[[i]]$forecast)] <- 0;
                yForecastCombined[] <- yForecastCombined + modelReturned$models[[i]]$forecast * adamSelected$icWeights[i];
            }

            # Prepare the name of the model
            modelName <- "";
            if(xregModel){
                modelName[] <- "ETSX";
            }
            else{
                modelName[] <- "ETS";
            }
            modelName[] <- paste0(modelName,"(",model,")");
            if(all(occurrence!=c("n","none"))){
                modelName[] <- paste0("i",modelName);
            }
            if(componentsNumberETSSeasonal>1){
                modelName[] <- paste0(modelName,"[",paste0(lags[lags!=1], collapse=", "),"]");
            }
            if(arimaModel){
                # Either the lags are non-seasonal, or there are no orders for seasonal lags
                if(all(lags==1) || (all(arOrders[lags>1]==0) && all(iOrders[lags>1]==0) && all(maOrders[lags>1]==0))){
                    modelName[] <- paste0(modelName,"+ARIMA(",arOrders[1],",",iOrders[1],",",maOrders[1],")");
                }
                else{
                    modelName[] <- paste0(modelName,"+SARIMA");
                    for(i in 1:length(arOrders)){
                        if(all(arOrders[i]==0) && all(iOrders[i]==0) && all(maOrders[i]==0)){
                            next;
                        }
                        modelName[] <- paste0(modelName,"(",arOrders[i],",");
                        modelName[] <- paste0(modelName,iOrders[i],",");
                        modelName[] <- paste0(modelName,maOrders[i],")[",lags[i],"]");
                    }
                }
            }
            if(all(occurrence!=c("n","none"))){
                modelName[] <- paste0("i",modelName);
            }

            modelReturned$models[[i]]$model <- modelName;
            modelReturned$models[[i]]$timeElapsed <- Sys.time()-startTime;
            parametersNumberOverall[1,1] <- parametersNumber[1,1] + parametersNumber[1,1] * adamSelected$icWeights[i];
            if(!is.null(xregData)){
                modelReturned$models[[i]]$data <- data[1:obsInSample,,drop=FALSE];
                if(holdout){
                    modelReturned$models[[i]]$holdout <- data[obsInSample+c(1:h),,drop=FALSE];
                }
                # Fix the ts class, which is destroyed during subsetting
                if(all(yClasses!="zoo")){
                    if(is.data.frame(data)){
                        modelReturned$models[[i]]$data[,responseName] <- ts(modelReturned$models[[i]]$data[,responseName],
                                                                            start=yStart, frequency=yFrequency);
                        if(holdout){
                            modelReturned$models[[i]]$holdout[,responseName] <- ts(modelReturned$models[[i]]$holdout[,responseName],
                                                                                   start=yForecastStart, frequency=yFrequency);
                        }
                    }
                    else{
                        modelReturned$models[[i]]$data <- ts(modelReturned$models[[i]]$data, start=yStart, frequency=yFrequency);
                        if(holdout){
                            modelReturned$models[[i]]$holdout <- ts(modelReturned$models[[i]]$holdout, start=yForecastStart, frequency=yFrequency);
                        }
                    }
                }
            }
            else{
                modelReturned$models[[i]]$data <- yInSample;
                modelReturned$models[[i]]$holdout <- yHoldout;
            }
            if(any(yNAValues)){
                modelReturned$models[[i]]$data[yNAValues[1:obsInSample],responseName] <- NA;
                if(length(yNAValues)==obsAll){
                    modelReturned$models[[i]]$holdout[yNAValues[-c(1:obsInSample)],responseName] <- NA;
                }
                modelReturned$models[[i]]$residuals[yNAValues[1:obsInSample]] <- NA;
            }
            modelReturned$models[[i]]$call <- cl;

            class(modelReturned$models[[i]]) <- c("adam","smooth");
        }

        # Record the original name of the model.
        model[] <- modelOriginal;
        # Prepare the name of the model
        modelName <- "";
        if(xregModel){
            modelName[] <- "ETSX";
        }
        else{
            modelName[] <- "ETS";
        }
        modelName[] <- paste0(modelName,"(",model,")");
        if(all(occurrence!=c("n","none"))){
            modelName[] <- paste0("i",modelName);
        }
        if(componentsNumberETSSeasonal>1){
            modelName[] <- paste0(modelName,"[",paste0(lags[lags!=1], collapse=", "),"]");
        }
        if(arimaModel){
            # Either the lags are non-seasonal, or there are no orders for seasonal lags
            if(all(lags==1) || (all(arOrders[lags>1]==0) && all(iOrders[lags>1]==0) && all(maOrders[lags>1]==0))){
                modelName[] <- paste0(modelName,"+ARIMA(",arOrders[1],",",iOrders[1],",",maOrders[1],")");
            }
            else{
                modelName[] <- paste0(modelName,"+SARIMA");
                for(i in 1:length(arOrders)){
                    if(all(arOrders[i]==0) && all(iOrders[i]==0) && all(maOrders[i]==0)){
                        next;
                    }
                    modelName[] <- paste0(modelName,"(",arOrders[i],",");
                    modelName[] <- paste0(modelName,iOrders[i],",");
                    modelName[] <- paste0(modelName,maOrders[i],")[",lags[i],"]");
                }
            }
        }
        if(all(occurrence!=c("n","none"))){
            modelName[] <- paste0("i",modelName);
        }
        modelReturned$model <- modelName;
        modelReturned$formula <- as.formula(paste0(responseName,"~."));
        modelReturned$timeElapsed <- Sys.time()-startTime;
        if(!is.null(xregDataOriginal)){
            modelReturned$data <- data[1:obsInSample,,drop=FALSE];
            if(holdout){
                modelReturned$holdout <- data[obsInSample+c(1:h),,drop=FALSE];
            }
            # Fix the ts class, which is destroyed during subsetting
            if(all(yClasses!="zoo")){
                if(is.data.frame(data)){
                    modelReturned$data[,responseName] <- ts(modelReturned$data[,responseName],
                                                            start=yStart, frequency=yFrequency);
                    if(holdout){
                        modelReturned$holdout[,responseName] <- ts(modelReturned$holdout[,responseName],
                                                                   start=yForecastStart, frequency=yFrequency);
                    }
                }
                else{
                    modelReturned$data <- ts(modelReturned$data, start=yStart, frequency=yFrequency);
                    if(holdout){
                        modelReturned$holdout <- ts(modelReturned$holdout, start=yForecastStart, frequency=yFrequency);
                    }
                }
            }
        }
        else{
            modelReturned$data <- yInSample;
            modelReturned$holdout <- yHoldout;
        }
        modelReturned$fitted <- ts(yFittedCombined,start=yStart, frequency=yFrequency);
        modelReturned$residuals <- yInSample - yFittedCombined;
        if(any(yNAValues)){
            modelReturned$data[yNAValues[1:obsInSample],responseName] <- NA;
            if(length(yNAValues)==obsAll){
                modelReturned$holdout[yNAValues[-c(1:obsInSample)],responseName] <- NA;
            }
            modelReturned$residuals[yNAValues[1:obsInSample]] <- NA;
        }
        modelReturned$forecast <- ts(yForecastCombined,start=yForecastStart, frequency=yFrequency);
        parametersNumberOverall[1,4] <- sum(parametersNumberOverall[1,1:3]);
        parametersNumberOverall[2,4] <- sum(parametersNumberOverall[2,1:3]);
        modelReturned$nParam <- parametersNumberOverall;
        modelReturned$ICw <- adamSelected$icWeights;
        # These two are needed just to make basic methods work
        modelReturned$distribution <- distribution;
        modelReturned$scale <- sqrt(mean(modelReturned$residuals^2,na.rm=TRUE));
        class(modelReturned) <- c("adamCombined","adam","smooth");
    }
    modelReturned$ICs <- icSelection;
    modelReturned$lossFunction <- lossFunction;
    modelReturned$call <- cl;
    modelReturned$bounds <- bounds;

    # Error measures if there is a holdout
    if(holdout){
        modelReturned$accuracy <- measures(yHoldout,modelReturned$forecast,yInSample);
    }

    if(!silent){
        plot(modelReturned, 7);
    }

    return(modelReturned);
}

#### Small useful ADAM functions ####
# This function creates recent and observed profiles for adam
#' @importFrom greybox detectdst
adamProfileCreator <- function(lagsModelAll, lagsModelMax, obsAll,
                               lags=NULL, yIndex=NULL, yClasses=NULL){
    # lagsModelAll - all lags used in the model for ETS + ARIMA + xreg
    # lagsModelMax - the maximum lag used in the model
    # obsAll - number of observations to create
    # lags - the original lags provided by user (no lags for ARIMA etc). Needed in order to see
    #        if weird frequencies are used.
    # yIndex - the indices needed in order to get the weird dates.
    # yClass - the class used for the actuals. If zoo, magic will happen here.
    # Create the matrix with profiles, based on provided lags
    profilesRecentTable <- matrix(0,length(lagsModelAll),lagsModelMax,
                                  dimnames=list(lagsModelAll,NULL));
    # Create the matrix of observed profiles indices
    profilesObservedTable <- matrix(1,length(lagsModelAll),obsAll,
                                    dimnames=list(lagsModelAll,NULL));
    # Modify the observed ones in order to get proper indices in C++
    profileIndices <- matrix(c(1:(lagsModelMax*length(lagsModelAll))),length(lagsModelAll));

    for(i in 1:length(lagsModelAll)){
        profilesRecentTable[i,1:lagsModelAll[i]] <- 1:lagsModelAll[i];
        # -1 is needed to align this with C++ code
        profilesObservedTable[i,] <- rep(profileIndices[i,1:lagsModelAll[i]],ceiling(obsAll/lagsModelAll[i]))[1:obsAll] -1;
    }

    # Do shifts for proper lags only:
    # Check lags variable for 24 / 24*7 / 24*365 / 48 / 48*7 / 48*365 / 365 / 52
    # If they are there, find the DST / Leap moments
    # Then amend respective observed values of profile, shifting them around
    if(any(yClasses=="zoo") && !is.null(yIndex)){
        # If this is weekly data, duplicate 52, when 53 is used
        if(any(lags==52) && any(strftime(yIndex,format="%W")=="53")){
            shiftRows <- lagsModelAll==52;
            # If the data does not start with 1, proceed
            if(all(which(strftime(yIndex,format="%W")=="53")!=1)){
                profilesObservedTable[shiftRows,which(strftime(yIndex,format="%W")=="53")] <-
                    profilesObservedTable[shiftRows,which(strftime(yIndex,format="%W")=="53")-1];
            }
        }

        #### If this is daily and we have 365 days of year, locate 29th February and use 28th instead
        if(any(c(365,365*48,365*24) %in% lags) && any(strftime(yIndex,format="%d/%m")=="29/02")){
            shiftValue <- c(365,365*48,365*24)[c(365,365*48,365*24) %in% lags]/365;
            shiftRows <- lagsModelAll %in% c(365,365*48,365*24);
            # If the data does not start with 1/24/48, proceed (otherwise we refer to negative numbers)
            if(!any(which(strftime(yIndex,format="%d/%m")=="29/02") %in% shiftValue)){
                profilesObservedTable[shiftRows,which(strftime(yIndex,format="%d/%m")=="29/02")] <-
                    profilesObservedTable[shiftRows,which(strftime(yIndex,format="%d/%m")=="29/02")-shiftValue];
            }
        }

        #### If this is hourly; Locate DST and do shifts for specific observations
        if(any(c(24,24*7,24*365,48,48*7,48*365) %in% lags)){
            shiftRows <- lagsModelAll %in% c(24,48,24*7,48*7,24*365,48*365);
            # If this is hourly data, then shift 1 hour. If it is halfhourly, shift 2 hours
            shiftValue <- 1;
            if(any(c(48,48*7,48*365) %in% lags)){
                shiftValue[] <- 2;
            }
            # Get the start and the end of DST
            dstValues <- detectdst(yIndex);
            # If the start date is not positioned before the end, introduce the artificial one
            if(nrow(dstValues$start)==0 ||
               (nrow(dstValues$end)>0 && dstValues$start$id[1]>dstValues$end$id[1])){
                dstValues$start <- rbind(data.frame(id=1,date=yIndex[1]),dstValues$start);
            }
            # If the end date is not present or the length of the end is not the same as the start,
            # set the end of series as one
            if(nrow(dstValues$end)==0 ||
               nrow(dstValues$end)<nrow(dstValues$start)){
                dstValues$end <- rbind(dstValues$end,data.frame(id=obsAll,date=tail(yIndex,1)));
            }
            # Shift everything from start to end dates by 1 obs forward.
            for(i in 1:nrow(dstValues$start)){
                # If the end date is natural, just shift
                if(dstValues$end$id[i]+shiftValue<=obsAll){
                    profilesObservedTable[shiftRows,dstValues$start$id[i]:dstValues$end$id[i]] <-
                        profilesObservedTable[shiftRows,dstValues$start$id[i]:dstValues$end$id[i]+shiftValue];
                }
                # If it isn't, we need to come up with the values for the end of sample
                else{
                    profilesObservedTable[shiftRows,dstValues$start$id[i]:dstValues$end$id[i]] <-
                        profilesObservedTable[shiftRows,dstValues$start$id[i]:dstValues$end$id[i]-lagsModelMax+shiftValue];
                }
            }
        }
    }

    return(list(recent=profilesRecentTable,observed=profilesObservedTable));
}

#### ARI and MA polynomials function ####
polynomialiser <- function(B, arOrders, iOrders, maOrders,
                           arRequired, maRequired, arEstimate, maEstimate, armaParameters, lags){

    # Number of parameters that we have
    nParamAR <- sum(arOrders);
    nParamMA <- sum(maOrders);

    # Matrices with parameters
    arParameters <- matrix(0, max(arOrders * lags) + 1, length(arOrders));
    iParameters <- matrix(0, max(iOrders * lags) + 1, length(iOrders));
    maParameters <- matrix(0, max(maOrders * lags) + 1, length(maOrders));
    # The first element is always 1
    arParameters[1,] <- iParameters[1,] <- maParameters[1,] <- 1;

    # nParam is used for B
    nParam <- 1;
    # armanParam is used for the provided arma parameters
    armanParam <- 1;
    # Fill in the matrices with the provided parameters
    for(i in 1:length(lags)){
        if(arOrders[i]*lags[i]!=0){
            if(arEstimate){
                arParameters[1+(1:arOrders[i])*lags[i],i] <- -B[nParam+c(1:arOrders[i])-1];
                nParam <- nParam + arOrders[i];
            }
            else if(!arEstimate && arRequired){
                arParameters[1+(1:arOrders[i])*lags[i],i] <- -armaParameters[armanParam+c(1:arOrders[i])-1];
                armanParam <- armanParam + arOrders[i];
            }
        }

        if(iOrders[i]*lags[i] != 0){
            iParameters[1+lags[i],i] <- -1;
        }

        if(maOrders[i]*lags[i]!=0){
            if(maEstimate){
                maParameters[1+(1:maOrders[i])*lags[i],i] <- B[nParam+c(1:maOrders[i])-1];
                nParam <- nParam + maOrders[i];
            }
            else if(!maEstimate && maRequired){
                maParameters[1+(1:maOrders[i])*lags[i],i] <- armaParameters[armanParam+c(1:maOrders[i])-1];
                armanParam <- armanParam + maOrders[i];
            }
        }
    }

    # Vectors of polynomials for the ARIMA
    arPolynomial <- vector("numeric", sum(arOrders * lags) + 1);
    iPolynomial <- vector("numeric", sum(iOrders * lags) + 1);
    maPolynomial <- vector("numeric", sum(maOrders * lags) + 1);
    ariPolynomial <- vector("numeric", sum(arOrders * lags) + sum(iOrders * lags) + 1);

    # Fill in the first polynomials
    arPolynomial[0:(arOrders[1]*lags[1])+1] <- arParameters[0:(arOrders[1]*lags[1])+1,1];
    iPolynomial[0:(iOrders[1]*lags[1])+1] <- iParameters[0:(iOrders[1]*lags[1])+1,1];
    maPolynomial[0:(maOrders[1]*lags[1])+1] <- maParameters[0:(maOrders[1]*lags[1])+1,1];

    index1 <- 0
    index2 <- 0;
    # Fill in all the other polynomials
    for(i in 1:length(lags)){
        if(i!=1){
            if(arOrders[i]>0){
                index1[] <- tail(which(arPolynomial!=0),1);
                index2[] <- tail(which(arParameters[,i]!=0),1);
                arPolynomial[1:(index1+index2-1)] <- polyprod(arPolynomial[1:index1], arParameters[1:index2,i]);
            }

            if(maOrders[i]>0){
                index1[] <- tail(which(maPolynomial!=0),1);
                index2[] <- tail(which(maParameters[,i]!=0),1);
                maPolynomial[1:(index1+index2-1)] <- polyprod(maPolynomial[1:index1], maParameters[1:index2,i]);
            }

            if(iOrders[i]>0){
                index1[] <- tail(which(iPolynomial!=0),1);
                index2[] <- tail(which(iParameters[,i]!=0),1);
                iPolynomial[1:(index1+index2-1)] <- polyprod(iPolynomial[1:index1], iParameters[1:index2,i]);
            }
        }
        # This part takes the power of (1-B)^D
        if(iOrders[i]>1){
            for(j in 2:iOrders[i]){
                index1[] <- tail(which(iPolynomial!=0),1);
                index2[] <- tail(which(iParameters[,i]!=0),1);
                iPolynomial[1:(index1+index2-1)] = polyprod(iPolynomial[1:index1], iParameters[1:index2,i]);
            }
        }
    }
    # ARI polynomials
    ariPolynomial[] <- polyprod(arPolynomial, iPolynomial);

    return(list(arPolynomial=arPolynomial,iPolynomial=iPolynomial,
                ariPolynomial=ariPolynomial,maPolynomial=maPolynomial));
}

#### Technical methods ####
#' @export
lags.adam <- function(object, ...){
    return(object$lags);
}

#' @rdname plot.smooth
#' @export
plot.adam <- function(x, which=c(1,2,4,6), level=0.95, legend=FALSE,
                      ask=prod(par("mfcol")) < length(which) && dev.interactive(),
                      lowess=TRUE, ...){
    ellipsis <- list(...);

    # Define, whether to wait for the hit of "Enter"
    if(ask){
        oask <- devAskNewPage(TRUE);
        on.exit(devAskNewPage(oask));
    }

    # 1. Fitted vs Actuals values
    plot1 <- function(x, ...){
        ellipsis <- list(...);

        # Get the actuals and the fitted values
        ellipsis$y <- as.vector(actuals(x));
        if(is.occurrence(x)){
            if(any(x$distribution==c("plogis","pnorm")) || x$occurrence!="none"){
                ellipsis$y <- (ellipsis$y!=0)*1;
            }
        }
        ellipsis$x <- as.vector(fitted(x));

        # If this is a mixture model, remove zeroes
        if(is.occurrence(x$occurrence)){
            ellipsis$x <- ellipsis$x[ellipsis$y!=0];
            ellipsis$y <- ellipsis$y[ellipsis$y!=0];
        }

        # Remove NAs
        if(any(is.na(ellipsis$x))){
            ellipsis$y <- ellipsis$y[!is.na(ellipsis$x)];
            ellipsis$x <- ellipsis$x[!is.na(ellipsis$x)];
        }
        if(any(is.na(ellipsis$y))){
            ellipsis$x <- ellipsis$x[!is.na(ellipsis$y)];
            ellipsis$y <- ellipsis$y[!is.na(ellipsis$y)];
        }

        # Title
        if(!any(names(ellipsis)=="main")){
            ellipsis$main <- "Actuals vs Fitted";
        }
        # If type and ylab are not provided, set them...
        if(!any(names(ellipsis)=="type")){
            ellipsis$type <- "p";
        }
        if(!any(names(ellipsis)=="ylab")){
            ellipsis$ylab <- "Actuals";
        }
        if(!any(names(ellipsis)=="xlab")){
            ellipsis$xlab <- "Fitted";
        }
        # xlim and ylim
        if(!any(names(ellipsis)=="xlim")){
            ellipsis$xlim <- range(c(ellipsis$x,ellipsis$y));
        }
        if(!any(names(ellipsis)=="ylim")){
            ellipsis$ylim <- range(c(ellipsis$x,ellipsis$y));
        }

        # Start plotting
        do.call(plot,ellipsis);
        abline(a=0,b=1,col="grey",lwd=2,lty=2)
        if(lowess){
            lines(lowess(ellipsis$x, ellipsis$y), col="red");
        }
    }

    # 2 and 3: Standardised  / studentised residuals vs Fitted
    plot2 <- function(x, type="rstandard", ...){
        ellipsis <- list(...);

        ellipsis$x <- as.vector(fitted(x));
        if(type=="rstandard"){
            ellipsis$y <- as.vector(rstandard(x));
            yName <- "Standardised";
        }
        else{
            ellipsis$y <- as.vector(rstudent(x));
            yName <- "Studentised";
        }

        if(is.occurrence(x$occurrence)){
            ellipsis$x <- ellipsis$x[actuals(x$occurrence)!=0];
            ellipsis$y <- ellipsis$y[actuals(x$occurrence)!=0];
        }

        # Remove NAs
        if(any(is.na(ellipsis$x))){
            ellipsis$x <- ellipsis$x[!is.na(ellipsis$x)];
            ellipsis$y <- ellipsis$y[!is.na(ellipsis$y)];
        }

        # Main, labs etc
        if(!any(names(ellipsis)=="main")){
            if(any(x$distribution==c("dinvgauss","dlnorm","dllaplace","dls","dlgnorm"))){
                ellipsis$main <- paste0("log(",yName," Residuals) vs Fitted");
            }
            else{
                ellipsis$main <- paste0(yName," Residuals vs Fitted");
            }
        }

        if(!any(names(ellipsis)=="xlab")){
            ellipsis$xlab <- "Fitted";
        }
        if(!any(names(ellipsis)=="ylab")){
            ellipsis$ylab <- paste0(yName," Residuals");
        }

        if(legend){
            if(ellipsis$x[length(ellipsis$x)]>mean(ellipsis$x)){
                legendPosition <- "bottomright";
            }
            else{
                legendPosition <- "topright";
            }
        }

        zValues <- switch(x$distribution,
                          "dlaplace"=,
                          "dllaplace"=qlaplace(c((1-level)/2, (1+level)/2), 0, 1),
                          "dalaplace"=qalaplace(c((1-level)/2, (1+level)/2), 0, 1, x$other$alpha),
                          "dlogis"=qlogis(c((1-level)/2, (1+level)/2), 0, 1),
                          "dt"=qt(c((1-level)/2, (1+level)/2), nobs(x)-nparam(x)),
                          "ds"=,
                          "dls"=qs(c((1-level)/2, (1+level)/2), 0, 1),
                          "dgnorm"=,
                          "dlgnorm"=qgnorm(c((1-level)/2, (1+level)/2), 0, 1, x$other$beta),
                          # In the next one, the scale is debiased, taking n-k into account
                          "dinvgauss"=qinvgauss(c((1-level)/2, (1+level)/2), mean=1,
                                                dispersion=x$scale * nobs(x) / (nobs(x)-nparam(x))),
                          "dlnorm"=qlnorm(c((1-level)/2, (1+level)/2), 0, 1),
                          qnorm(c((1-level)/2, (1+level)/2), 0, 1));
        # Analyse stuff in logarithms if the error is multiplicative
        if(any(x$distribution==c("dinvgauss","dlnorm"))){
            ellipsis$y[] <- log(ellipsis$y);
            zValues <- log(zValues);
        }
        else if(any(x$distribution==c("dllaplace","dls","dlgnorm"))){
            ellipsis$y[] <- log(ellipsis$y);
        }
        outliers <- which(ellipsis$y >zValues[2] | ellipsis$y <zValues[1]);
        # cat(paste0(round(length(outliers)/length(ellipsis$y),3)*100,"% of values are outside the bounds\n"));


        if(!any(names(ellipsis)=="ylim")){
            ellipsis$ylim <- range(c(ellipsis$y,zValues), na.rm=TRUE)*1.2;
            if(legend){
                if(legendPosition=="bottomright"){
                    ellipsis$ylim[1] <- ellipsis$ylim[1] - 0.2*diff(ellipsis$ylim);
                }
                else{
                    ellipsis$ylim[2] <- ellipsis$ylim[2] + 0.2*diff(ellipsis$ylim);
                }
            }
        }

        xRange <- range(ellipsis$x, na.rm=TRUE);
        xRange[1] <- xRange[1] - sd(ellipsis$x, na.rm=TRUE);
        xRange[2] <- xRange[2] + sd(ellipsis$x, na.rm=TRUE);

        do.call(plot,ellipsis);
        abline(h=0, col="grey", lty=2);
        polygon(c(xRange,rev(xRange)),c(zValues[1],zValues[1],zValues[2],zValues[2]),
                col="lightgrey", border=NA, density=10);
        abline(h=zValues, col="red", lty=2);
        if(length(outliers)>0){
            points(ellipsis$x[outliers], ellipsis$y[outliers], pch=16);
            text(ellipsis$x[outliers], ellipsis$y[outliers], labels=outliers, pos=(ellipsis$y[outliers]>0)*2+1);
        }
        if(lowess){
            lines(lowess(ellipsis$x[!is.na(ellipsis$y)], ellipsis$y[!is.na(ellipsis$y)]), col="red");
        }

        if(legend){
            if(lowess){
                legend(legendPosition,
                       legend=c(paste0(round(level,3)*100,"% bounds"),"outside the bounds","LOWESS line"),
                       col=c("red", "black","red"), lwd=c(1,NA,1), lty=c(2,1,1), pch=c(NA,16,NA));
            }
            else{
                legend(legendPosition,
                       legend=c(paste0(round(level,3)*100,"% bounds"),"outside the bounds"),
                       col=c("red", "black"), lwd=c(1,NA), lty=c(2,1), pch=c(NA,16));
            }
        }
    }

    # 4 and 5. Fitted vs |Residuals| or Fitted vs Residuals^2
    plot3 <- function(x, type="abs", ...){
        ellipsis <- list(...);

        ellipsis$x <- as.vector(fitted(x));
        ellipsis$y <- as.vector(residuals(x));
        if(any(x$distribution==c("dinvgauss","dlnorm","dllaplace","dls","dlgnorm"))){
            ellipsis$y[] <- log(ellipsis$y);
        }
        if(type=="abs"){
            ellipsis$y[] <- abs(ellipsis$y);
        }
        else{
            ellipsis$y[] <- as.vector(ellipsis$y)^2;
        }

        if(is.occurrence(x$occurrence)){
            ellipsis$x <- ellipsis$x[ellipsis$y!=0];
            ellipsis$y <- ellipsis$y[ellipsis$y!=0];
        }
        # Remove NAs
        if(any(is.na(ellipsis$x))){
            ellipsis$x <- ellipsis$x[!is.na(ellipsis$x)];
            ellipsis$y <- ellipsis$y[!is.na(ellipsis$y)];
        }

        if(!any(names(ellipsis)=="main")){
            if(type=="abs"){
                if(any(x$distribution==c("dinvgauss","dlnorm","dllaplace","dls","dlgnorm"))){
                    ellipsis$main <- "|log(Residuals)| vs Fitted";
                }
                else{
                    ellipsis$main <- "|Residuals| vs Fitted";
                }
            }
            else{
                if(any(x$distribution==c("dinvgauss","dlnorm","dllaplace","dls","dlgnorm"))){
                    ellipsis$main <- "log(Residuals)^2 vs Fitted";
                }
                else{
                    ellipsis$main <- "Residuals^2 vs Fitted";
                }
            }
        }

        if(!any(names(ellipsis)=="xlab")){
            ellipsis$xlab <- "Fitted";
        }
        if(!any(names(ellipsis)=="ylab")){
            if(type=="abs"){
                ellipsis$ylab <- "|Residuals|";
            }
            else{
                ellipsis$ylab <- "Residuals^2";
            }
        }

        do.call(plot,ellipsis);
        abline(h=0, col="grey", lty=2);
        if(lowess){
            lines(lowess(ellipsis$x[!is.na(ellipsis$y)], ellipsis$y[!is.na(ellipsis$y)]), col="red");
        }
    }

    # 6. Q-Q with the specified distribution
    plot4 <- function(x, ...){
        ellipsis <- list(...);

        ellipsis$y <- as.vector(residuals(x));
        if(is.occurrence(x$occurrence)){
            ellipsis$y <- ellipsis$y[actuals(x$occurrence)!=0];
        }

        if(!any(names(ellipsis)=="xlab")){
            ellipsis$xlab <- "Theoretical Quantile";
        }
        if(!any(names(ellipsis)=="ylab")){
            ellipsis$ylab <- "Actual Quantile";
        }

        if(any(x$distribution=="dnorm")){
            if(!any(names(ellipsis)=="main")){
                ellipsis$main <- "QQ plot of Normal distribution";
            }

            do.call(qqnorm, ellipsis);
            qqline(ellipsis$y);
        }
        else if(any(x$distribution=="dlnorm")){
            if(!any(names(ellipsis)=="main")){
                ellipsis$main <- "QQ plot of Log Normal distribution";
            }
            ellipsis$x <- qlnorm(ppoints(500), meanlog=0, sdlog=x$scale);

            do.call(qqplot, ellipsis);
            qqline(ellipsis$y, distribution=function(p) qlnorm(p, meanlog=0, sdlog=x$scale));
        }
        else if(x$distribution=="dlaplace"){
            if(!any(names(ellipsis)=="main")){
                ellipsis$main <- "QQ-plot of Laplace distribution";
            }
            ellipsis$x <- qlaplace(ppoints(500), mu=0, scale=x$scale);

            do.call(qqplot, ellipsis);
            qqline(ellipsis$y, distribution=function(p) qlaplace(p, mu=0, scale=x$scale));
        }
        else if(x$distribution=="dllaplace"){
            if(!any(names(ellipsis)=="main")){
                ellipsis$main <- "QQ-plot of Log Laplace distribution";
            }
            ellipsis$x <- exp(qlaplace(ppoints(500), mu=0, scale=x$scale));

            do.call(qqplot, ellipsis);
            qqline(ellipsis$y, distribution=function(p) exp(qlaplace(p, mu=0, scale=x$scale)));
        }
        else if(x$distribution=="ds"){
            if(!any(names(ellipsis)=="main")){
                ellipsis$main <- "QQ-plot of S distribution";
            }
            ellipsis$x <- qs(ppoints(500), mu=0, scale=x$scale);

            do.call(qqplot, ellipsis);
            qqline(ellipsis$y, distribution=function(p) qs(p, mu=0, scale=x$scale));
        }
        else if(x$distribution=="dls"){
            if(!any(names(ellipsis)=="main")){
                ellipsis$main <- "QQ-plot of Log S distribution";
            }
            ellipsis$x <- exp(qs(ppoints(500), mu=0, scale=x$scale));

            do.call(qqplot, ellipsis);
            qqline(ellipsis$y, distribution=function(p) exp(qs(p, mu=0, scale=x$scale)));
        }
        else if(x$distribution=="dgnorm"){
            if(!any(names(ellipsis)=="main")){
                ellipsis$main <- paste0("QQ-plot of Generalised Normal distribution with beta=",round(x$other$beta,3));
            }
            ellipsis$x <- qgnorm(ppoints(500), mu=0, alpha=x$scale, beta=x$other$beta);

            do.call(qqplot, ellipsis);
            qqline(ellipsis$y, distribution=function(p) qgnorm(p, mu=0, alpha=x$scale, beta=x$other$beta));
        }
        else if(x$distribution=="dlgnorm"){
            if(!any(names(ellipsis)=="main")){
                ellipsis$main <- paste0("QQ-plot of Log Generalised Normal distribution with beta=",round(x$other$beta,3));
            }
            ellipsis$x <- exp(qgnorm(ppoints(500), mu=0, alpha=x$scale, beta=x$other$beta));

            do.call(qqplot, ellipsis);
            qqline(ellipsis$y, distribution=function(p) exp(qgnorm(p, mu=0, alpha=x$scale, beta=x$other$beta)));
        }
        else if(x$distribution=="dlogis"){
            if(!any(names(ellipsis)=="main")){
                ellipsis$main <- "QQ-plot of Logistic distribution";
            }
            ellipsis$x <- qlogis(ppoints(500), location=0, scale=x$scale);

            do.call(qqplot, ellipsis);
            qqline(ellipsis$y, distribution=function(p) qlogis(p, location=0, scale=x$scale));
        }
        else if(x$distribution=="dt"){
            # Standardise residuals
            ellipsis$y[] <- ellipsis$y / sd(ellipsis$y);
            if(!any(names(ellipsis)=="main")){
                ellipsis$main <- "QQ-plot of Student's distribution";
            }
            ellipsis$x <- qt(ppoints(500), df=x$other$nu);

            do.call(qqplot, ellipsis);
            qqline(ellipsis$y, distribution=function(p) qt(p, df=x$other$nu));
        }
        else if(x$distribution=="dalaplace"){
            if(!any(names(ellipsis)=="main")){
                ellipsis$main <- paste0("QQ-plot of Asymmetric Laplace with alpha=",round(x$other$alpha,3));
            }
            ellipsis$x <- qalaplace(ppoints(500), mu=0, scale=x$scale, alpha=x$other$alpha);

            do.call(qqplot, ellipsis);
            qqline(ellipsis$y, distribution=function(p) qalaplace(p, mu=0, scale=x$scale, alpha=x$other$alpha));
        }
        else if(x$distribution=="dinvgauss"){
            # Transform residuals for something meaningful
            # This is not 100% accurate, because the dispersion should change as well as mean...
            if(!any(names(ellipsis)=="main")){
                ellipsis$main <- "QQ-plot of Inverse Gaussian distribution";
            }
            ellipsis$x <- qinvgauss(ppoints(500), mean=1, dispersion=x$scale);

            do.call(qqplot, ellipsis);
            qqline(ellipsis$y, distribution=function(p) qinvgauss(p, mean=1, dispersion=x$scale));
        }
    }

    # 7. Basic plot over time
    plot5 <- function(x, ...){
        ellipsis <- list(...);

        ellipsis$actuals <- actuals(x);
        if(!is.null(x$holdout)){
            if(is.zoo(ellipsis$actuals)){
                ellipsis$actuals <- zoo(c(as.vector(ellipsis$actuals),as.vector(x$holdout[,1])),
                                        order.by=c(time(ellipsis$actuals),time(x$holdout)));
            }
            else{
                ellipsis$actuals <- ts(c(ellipsis$actuals,x$holdout[,1]),
                                       start=start(ellipsis$actuals),
                                       frequency=frequency(ellipsis$actuals));
            }
        }
        if(is.null(ellipsis$main)){
            ellipsis$main <- x$model;
        }
        ellipsis$forecast <- x$forecast;
        ellipsis$fitted <- fitted(x);
        ellipsis$legend <- FALSE;
        ellipsis$parReset <- FALSE;

        do.call(graphmaker, ellipsis);
    }

    # 8 and 9. Standardised / Studentised residuals vs time
    plot6 <- function(x, type="rstandard", ...){

        ellipsis <- list(...);
        if(type=="rstandard"){
            ellipsis$x <- rstandard(x);
            yName <- "Standardised";
        }
        else{
            ellipsis$x <- rstudent(x);
            yName <- "Studentised";
        }

        if(is.occurrence(x$occurrence)){
            ellipsis$x <- ellipsis$x[actuals(x$occurrence)!=0];
        }

        if(!any(names(ellipsis)=="main")){
            ellipsis$main <- paste0(yName," Residuals vs Time");
        }

        if(!any(names(ellipsis)=="xlab")){
            ellipsis$xlab <- "Time";
        }
        if(!any(names(ellipsis)=="ylab")){
            ellipsis$ylab <- paste0(yName," Residuals");
        }

        # If type and ylab are not provided, set them...
        if(!any(names(ellipsis)=="type")){
            ellipsis$type <- "l";
        }

        zValues <- switch(x$distribution,
                          "dlnorm"=qlnorm(c((1-level)/2, (1+level)/2), 0, 1),
                          "dlaplace"=,
                          "dllaplace"=qlaplace(c((1-level)/2, (1+level)/2), 0, 1),
                          "ds"=,
                          "dls"=qs(c((1-level)/2, (1+level)/2), 0, 1),
                          "dgnorm"=,
                          "dlgnorm"=qgnorm(c((1-level)/2, (1+level)/2), 0, 1, x$other$beta),
                          "dlogis"=qlogis(c((1-level)/2, (1+level)/2), 0, 1),
                          "dt"=qt(c((1-level)/2, (1+level)/2), nobs(x)-nparam(x)),
                          "dalaplace"=qalaplace(c((1-level)/2, (1+level)/2), 0, 1, x$other$alpha),
                          # In the next one, the scale is debiased, taking n-k into account
                          "dinvgauss"=qinvgauss(c((1-level)/2, (1+level)/2), mean=1,
                                                dispersion=x$scale * nobs(x) / (nobs(x)-nparam(x))),
                          qnorm(c((1-level)/2, (1+level)/2), 0, 1));
        # Analyse stuff in logarithms if the error is multiplicative
        if(any(x$distribution==c("dinvgauss","dlnorm"))){
            ellipsis$x[] <- log(ellipsis$x);
            zValues <- log(zValues);
        }
        else if(any(x$distribution==c("dllaplace","dls","dlgnorm"))){
            ellipsis$x[] <- log(ellipsis$x);
        }
        outliers <- which(ellipsis$x >zValues[2] | ellipsis$x <zValues[1]);


        if(!any(names(ellipsis)=="ylim")){
            ellipsis$ylim <- c(-max(abs(ellipsis$x)),max(abs(ellipsis$x)))*1.2;
        }

        if(legend){
            legendPosition <- "topright";
            ellipsis$ylim[2] <- ellipsis$ylim[2] + 0.2*diff(ellipsis$ylim);
            ellipsis$ylim[1] <- ellipsis$ylim[1] - 0.2*diff(ellipsis$ylim);
        }

        # Start plotting
        do.call(plot,ellipsis);
        if(length(outliers)>0){
            points(time(ellipsis$x)[outliers], ellipsis$x[outliers], pch=16);
            text(time(ellipsis$x)[outliers], ellipsis$x[outliers], labels=outliers, pos=(ellipsis$x[outliers]>0)*2+1);
        }
        if(lowess){
            lines(lowess(c(1:length(ellipsis$x)),ellipsis$x), col="red");
        }
        abline(h=0, col="grey", lty=2);
        abline(h=zValues[1], col="red", lty=2);
        abline(h=zValues[2], col="red", lty=2);
        polygon(c(1:nobs(x), c(nobs(x):1)),
                c(rep(zValues[1],nobs(x)), rep(zValues[2],nobs(x))),
                col="lightgrey", border=NA, density=10);
        if(legend){
            legend(legendPosition,legend=c("Residuals",paste0(level*100,"% prediction interval")),
                   col=c("black","red"), lwd=rep(1,3), lty=c(1,1,2));
        }
    }

    # 10 and 11. ACF and PACF
    plot7 <- function(x, type="acf", ...){
        ellipsis <- list(...);

        if(!any(names(ellipsis)=="main")){
            if(type=="acf"){
                ellipsis$main <- "Autocorrelation Function of Residuals";
            }
            else{
                ellipsis$main <- "Partial Autocorrelation Function of Residuals";
            }
        }

        if(!any(names(ellipsis)=="xlab")){
            ellipsis$xlab <- "Lags";
        }
        if(!any(names(ellipsis)=="ylab")){
            if(type=="acf"){
                ellipsis$ylab <- "ACF";
            }
            else{
                ellipsis$ylab <- "PACF";
            }
        }

        if(!any(names(ellipsis)=="ylim")){
            ellipsis$ylim <- c(-1,1);
        }

        if(type=="acf"){
            theValues <- acf(as.vector(residuals(x)), plot=FALSE, na.action=na.pass);
        }
        else{
            theValues <- pacf(as.vector(residuals(x)), plot=FALSE, na.action=na.pass);
        }
        ellipsis$x <- theValues$acf[-1];
        zValues <- qnorm(c((1-level)/2, (1+level)/2),0,sqrt(1/nobs(x)));

        ellipsis$type <- "h"

        do.call(plot,ellipsis);
        abline(h=0, col="black", lty=1);
        abline(h=zValues, col="red", lty=2);
        if(any(ellipsis$x>zValues[2] | ellipsis$x<zValues[1])){
            outliers <- which(ellipsis$x >zValues[2] | ellipsis$x <zValues[1]);
            points(outliers, ellipsis$x[outliers], pch=16);
            text(outliers, ellipsis$x[outliers], labels=outliers, pos=(ellipsis$x[outliers]>0)*2+1);
        }
    }

    # 12. Plot of states
    plot8 <- function(x, ...){
        parDefault <- par(no.readonly = TRUE);
        if(any(unlist(gregexpr("C",x$model))==-1)){
            statesNames <- c("actuals",colnames(x$states),"residuals");
            x$states <- cbind(actuals(x),x$states,residuals(x));
            colnames(x$states) <- statesNames;
            if(ncol(x$states)>10){
                message("Too many states. Plotting them one by one on several graphs.");
                if(is.null(ellipsis$main)){
                    ellipsisMain <- NULL;
                }
                else{
                    ellipsisMain <- ellipsis$main;
                }
                nPlots <- ceiling(ncol(x$states)/10);
                for(i in 1:nPlots){
                    if(is.null(ellipsisMain)){
                        ellipsis$main <- paste0("States of ",x$model,", part ",i);
                    }
                    ellipsis$x <- x$states[,(1+(i-1)*10):min(i*10,ncol(x$states)),drop=FALSE];
                    do.call(plot, ellipsis);
                }
            }
            else{
                if(ncol(x$states)<=5){
                    ellipsis$nc <- 1;
                }
                if(is.null(ellipsis$main)){
                    ellipsis$main <- paste0("States of ",x$model);
                }
                ellipsis$x <- x$states;
                do.call(plot, ellipsis);
            }
        }
        else{
            # If we did combinations, we cannot return anything
            message("Combination of models was done. Sorry, but there is nothing to plot.");
        }
        par(parDefault);
    }

    # Do plots
    if(any(which==1)){
        plot1(x, ...);
    }

    if(any(which==2)){
        plot2(x, ...);
    }

    if(any(which==3)){
        plot2(x, "rstudent", ...);
    }

    if(any(which==4)){
        plot3(x, ...);
    }

    if(any(which==5)){
        plot3(x, type="squared", ...);
    }

    if(any(which==6)){
        plot4(x, ...);
    }

    if(any(which==7)){
        plot5(x, ...);
    }

    if(any(which==8)){
        plot6(x, ...);
    }

    if(any(which==9)){
        plot6(x, "rstudent", ...);
    }

    if(any(which==10)){
        plot7(x, type="acf", ...);
    }

    if(any(which==11)){
        plot7(x, type="pacf", ...);
    }

    if(any(which==12)){
        plot8(x, ...);
    }
}

#' @export
print.adam <- function(x, digits=4, ...){
    etsModel <- any(unlist(gregexpr("ETS",x$model))!=-1);
    arimaModel <- any(unlist(gregexpr("ARIMA",x$model))!=-1);

    cat("Time elapsed:",round(as.numeric(x$timeElapsed,units="secs"),2),"seconds");
    cat(paste0("\nModel estimated using ",x$call[[1]],"() function: ",x$model));

    if(is.occurrence(x$occurrence)){
        occurrence <- switch(x$occurrence$occurrence,
                             "f"=,
                             "fixed"="Fixed probability",
                             "o"=,
                             "odds-ratio"="Odds ratio",
                             "i"=,
                             "inverse-odds-ratio"="Inverse odds ratio",
                             "d"=,
                             "direct"="Direct",
                             "g"=,
                             "general"="General",
                             "p"=,
                             "provided"="Provided by user");
        cat("\nOccurrence model type:",occurrence);
    }

    distrib <- switch(x$distribution,
                      "dnorm" = "Normal",
                      "dlaplace" = "Laplace",
                      "ds" = "S",
                      "dgnorm" = paste0("Generalised Normal with beta=",round(x$other$beta, digits)),
                      "dlogis" = "Logistic",
                      "dt" = paste0("Student t with nu=",round(x$other$nu, digits)),
                      "dalaplace" = paste0("Asymmetric Laplace with alpha=",round(x$other$alpha,digits)),
                      "dlnorm" = "Log Normal",
                      "dllaplace" = "Log Laplace",
                      "dls" = "Log S",
                      "dlgnorm" = paste0("Log Generalised Normal with beta=",round(x$other$beta, digits)),
                      # "dbcnorm" = paste0("Box-Cox Normal with lambda=",round(x$other$lambda,2)),
                      "dinvgauss" = "Inverse Gaussian"
    );
    if(is.occurrence(x$occurrence)){
        distrib <- paste0("Mixture of Bernoulli and ", distrib);
    }
    cat("\nDistribution assumed in the model:", distrib);

    cat("\nLoss function type:",x$loss);
    if(!is.null(x$lossValue)){
        cat("; Loss function value:",round(x$lossValue,digits));
        if(any(x$loss==c("LASSO","RIDGE"))){
            cat("; lambda=",x$other$lambda);
        }
    }

    if(etsModel){
        if(!is.null(x$persistence)){
            cat("\nPersistence vector g");
            if(ncol(x$data)>1){
                cat(" (excluding xreg):\n");
            }
            else{
                cat(":\n");
            }
            persistence <- x$persistence[substr(names(x$persistence),1,5)!="delta"];
            if(arimaModel){
                persistence <- persistence[substr(names(persistence),1,3)!="psi"];
            }
            print(round(persistence,digits));
        }

        if(!is.null(x$phi)){
            if(gregexpr("d",x$model)!=-1){
                cat("Damping parameter:", round(x$phi,digits));
            }
        }
    }

    # If this is ARIMA model
    if(!is.null(x$arma) && (!is.null(x$arma$ar) || !is.null(x$arma$ma))){
        cat("\nARMA parameters of the model:\n");
        if(!is.null(x$arma$ar)){
            cat("AR:\n")
            print(round(x$arma$ar,digits));
        }
        if(!is.null(x$arma$ma)){
            cat("MA:\n")
            print(round(x$arma$ma,digits));
        }
    }

    cat("\nSample size:", nobs(x));
    cat("\nNumber of estimated parameters:", nparam(x));
    cat("\nNumber of degrees of freedom:", nobs(x)-nparam(x));
    if(x$nParam[2,4]>0){
        cat("\nNumber of provided parameters:", x$nParam[2,4]);
    }

    if(x$loss=="likelihood" ||
       (any(x$loss==c("MSE","MSEh","MSCE","GPL")) & (x$distribution=="dnorm")) ||
       (any(x$loss==c("aMSE","aMSEh","aMSCE","aGPL")) & (x$distribution=="dnorm")) ||
       (any(x$loss==c("MAE","MAEh","MACE")) & (x$distribution=="dlaplace")) ||
       (any(x$loss==c("HAM","HAMh","CHAM")) & (x$distribution=="ds"))){
        ICs <- c(AIC(x),AICc(x),BIC(x),BICc(x));
        names(ICs) <- c("AIC","AICc","BIC","BICc");
        cat("\nInformation criteria:\n");
        print(round(ICs,digits));
    }
    else{
        cat("\nInformation criteria are unavailable for the chosen loss & distribution.\n");
    }

    # If there are accuracy measures, print them out
    if(!is.null(x$accuracy)){
        cat("\nForecast errors:\n");
        if(is.null(x$occurrence)){
            cat(paste(paste0("ME: ",round(x$accuracy["ME"],3)),
                      paste0("MAE: ",round(x$accuracy["MAE"],3)),
                      paste0("RMSE: ",round(sqrt(x$accuracy["MSE"]),3),"\n")
                      # paste0("Bias: ",round(x$accuracy["cbias"],3)*100,"%"),
                      ,sep="; "));
            cat(paste(paste0("sCE: ",round(x$accuracy["sCE"],5)*100,"%"),
                      paste0("sMAE: ",round(x$accuracy["sMAE"],5)*100,"%"),
                      paste0("sMSE: ",round(x$accuracy["sMSE"],5)*100,"%\n")
                      ,sep="; "));
            cat(paste(paste0("MASE: ",round(x$accuracy["MASE"],3)),
                      paste0("RMSSE: ",round(x$accuracy["RMSSE"],3)),
                      paste0("rMAE: ",round(x$accuracy["rMAE"],3)),
                      paste0("rRMSE: ",round(x$accuracy["rRMSE"],3),"\n")
                      ,sep="; "));
        }
        else{
            cat(paste(paste0("Bias: ",round(x$accuracy["cbias"],5)*100,"%"),
                      paste0("sMSE: ",round(x$accuracy["sMSE"],5)*100,"%"),
                      paste0("rRMSE: ",round(x$accuracy["rRMSE"],3)),
                      paste0("sPIS: ",round(x$accuracy["sPIS"],5)*100,"%"),
                      paste0("sCE: ",round(x$accuracy["sCE"],5)*100,"%\n"),sep="; "));
        }
    }
}

#' @export
print.adamCombined <- function(x, digits=4, ...){
    cat("Time elapsed:",round(as.numeric(x$timeElapsed,units="secs"),2),"seconds");
    cat("\nModel estimated:",x$model);
    cat("\nLoss function type:",x$models[[1]]$loss);

    cat("\n\nNumber of models combined:", length(x$ICw));
    cat("\nSample size: "); cat(nobs(x));
    cat("\nAverage number of estimated parameters:", round(nparam(x),digits=digits));
    cat("\nAverage number of degrees of freedom:", round(nobs(x)-nparam(x),digits=digits));

    if(!is.null(x$accuracy)){
        cat("\n\nForecast errors:\n");
        if(is.null(x$occurrence)){
            cat(paste(paste0("ME: ",round(x$accuracy["ME"],3)),
                      paste0("MAE: ",round(x$accuracy["MAE"],3)),
                      paste0("RMSE: ",round(sqrt(x$accuracy["MSE"]),3),"\n")
                      # paste0("Bias: ",round(x$accuracy["cbias"],3)*100,"%"),
                      ,sep="; "));
            cat(paste(paste0("sCE: ",round(x$accuracy["sCE"],5)*100,"%"),
                      paste0("sMAE: ",round(x$accuracy["sMAE"],5)*100,"%"),
                      paste0("sMSE: ",round(x$accuracy["sMSE"],5)*100,"%\n")
                      ,sep="; "));
            cat(paste(paste0("MASE: ",round(x$accuracy["MASE"],3)),
                      paste0("RMSSE: ",round(x$accuracy["RMSSE"],3)),
                      paste0("rMAE: ",round(x$accuracy["rMAE"],3)),
                      paste0("rRMSE: ",round(x$accuracy["rRMSE"],3),"\n")
                      ,sep="; "));
        }
        else{
            cat(paste(paste0("Bias: ",round(x$accuracy["cbias"],5)*100,"%"),
                      paste0("sMSE: ",round(x$accuracy["sMSE"],5)*100,"%"),
                      paste0("rRMSE: ",round(x$accuracy["rRMSE"],3)),
                      paste0("sPIS: ",round(x$accuracy["sPIS"],5)*100,"%"),
                      paste0("sCE: ",round(x$accuracy["sCE"],5)*100,"%\n"),sep="; "));
        }
    }
}

#### Coefficients ####
#### The functions needed for confint and refit

# The function inverts the measurement matrix, setting infinite values to zero
# This is needed for the stability check for xreg models with regressors="adapt"
measurementInverter <- function(measurement){
    measurement[] <- 1/measurement;
    measurement[is.infinite(measurement)] <- 0;
    return(measurement);
}

# The function that returns the eigen values for specified parameters
# The function returns TRUE if the condition is violated
eigenValues <- function(object, persistence){
    #### !!!! Eigen values checks do not work for xreg. So move to (0, 1) region
    if(ncol(object$data)>1 && any(substr(names(object$persistence),1,5)=="delta")){
        # We check the condition on average
        return(any(abs(eigen((object$transition -
                                  diag(as.vector(persistence)) %*%
                                  t(measurementInverter(object$measurement[1:nobs(object),,drop=FALSE])) %*%
                                  object$measurement[1:nobs(object),,drop=FALSE] / nobs(object)),
                             symmetric=TRUE, only.values=TRUE)$values)>1+1E-10));
    }
    else{
        return(any(abs(eigen(object$transition -
                                 persistence %*% object$measurement[nobs(object),,drop=FALSE],
                             symmetric=TRUE, only.values=TRUE)$values)>1+1E-10));
    }
}

# The function that returns the bounds for persistence parameters, based on eigen values
eigenBounds <- function(object, persistence, variableNumber=1){
    # The lower bound
    persistence[variableNumber,] <- -5;
    eigenValuesTested <- eigenValues(object, persistence);
    while(eigenValuesTested){
        persistence[variableNumber,] <- persistence[variableNumber,] + 0.01;
        eigenValuesTested[] <- eigenValues(object, persistence);
        if(persistence[variableNumber,]>5){
            persistence[variableNumber,] <- -5;
            break;
        }
    }
    lowerBound <- persistence[variableNumber,]-0.01;
    # The upper bound
    persistence[variableNumber,] <- 5;
    eigenValuesTested <- eigenValues(object, persistence);
    while(eigenValuesTested){
        persistence[variableNumber,] <- persistence[variableNumber,] - 0.01;
        eigenValuesTested[] <- eigenValues(object, persistence);
        if(persistence[variableNumber,]<-5){
            persistence[variableNumber,] <- 5;
            break;
        }
    }
    upperBound <- persistence[variableNumber,]+0.01;
    return(c(lowerBound, upperBound));
}

# Function for the bounds of the AR parameters
arPolinomialsBounds <- function(arPolynomialMatrix,arPolynomial,variableNumber){
    # The lower bound
    arPolynomial[variableNumber] <- -5;
    arPolynomialMatrix[,1] <- -arPolynomial[-1];
    arPolyroots <- any(abs(eigen(arPolynomialMatrix, symmetric=TRUE, only.values=TRUE)$values)>1);
    while(arPolyroots){
        arPolynomial[variableNumber] <- arPolynomial[variableNumber] +0.01;
        arPolynomialMatrix[,1] <- -arPolynomial[-1];
        arPolyroots[] <- any(abs(eigen(arPolynomialMatrix, symmetric=TRUE, only.values=TRUE)$values)>1);
    }
    lowerBound <- arPolynomial[variableNumber]-0.01;
    # The upper bound
    arPolynomial[variableNumber] <- 5;
    arPolynomialMatrix[,1] <- -arPolynomial[-1];
    arPolyroots <- any(abs(eigen(arPolynomialMatrix, symmetric=TRUE, only.values=TRUE)$values)>1);
    while(arPolyroots){
        arPolynomial[variableNumber] <- arPolynomial[variableNumber] -0.01;
        arPolynomialMatrix[,1] <- -arPolynomial[-1];
        arPolyroots[] <- any(abs(eigen(arPolynomialMatrix, symmetric=TRUE, only.values=TRUE)$values)>1);
    }
    upperBound <- arPolynomial[variableNumber]+0.01;
    return(c(lowerBound, upperBound));
}


# Confidence intervals
#' @export
confint.adam <- function(object, parm, level=0.95, ...){
    adamVcov <- vcov(object, ...);
    parameters <- coef(object);
    adamSD <- sqrt(abs(diag(adamVcov)));
    parametersNames <- names(adamSD);
    nParam <- length(adamSD);
    etsModel <- any(unlist(gregexpr("ETS",object$model))!=-1);
    arimaModel <- any(unlist(gregexpr("ARIMA",object$model))!=-1);
    adamCoefBounds <- matrix(0,nParam,2,
                             dimnames=list(parametersNames,NULL));
    # Fill in the values with normal bounds
    adamCoefBounds[,1] <- qt((1-level)/2, df=nobs(object)-nparam(object))*adamSD;
    adamCoefBounds[,2] <- qt((1+level)/2, df=nobs(object)+nparam(object))*adamSD;

    persistence <- as.matrix(object$persistence);
    # If there is xreg, but no deltas, increase persistence by including zeroes
    if(ncol(object$data)>1 && !any(substr(names(object$persistence),1,5)=="delta")){
        persistence <- rbind(persistence,matrix(rep(0,sum(object$nParam[,2])),ncol=1));
    }

    # Correct the bounds for the ETS model
    if(etsModel){
        #### The usual bounds ####
        if(object$bounds=="usual"){
            # Check, if there is alpha
            if(any(parametersNames=="alpha")){
                adamCoefBounds["alpha",1] <- max(-parameters["alpha"],adamCoefBounds["alpha",1]);
                adamCoefBounds["alpha",2] <- min(1-parameters["alpha"],adamCoefBounds["alpha",2]);
            }
            # Check, if there is beta
            if(any(parametersNames=="beta")){
                adamCoefBounds["beta",1] <- max(-parameters["beta"],adamCoefBounds["beta",1]);
                if(any(parametersNames=="alpha")){
                    adamCoefBounds["beta",2] <- min(parameters["alpha"]-parameters["beta"],adamCoefBounds["beta",2]);
                }
                else{
                    adamCoefBounds["beta",2] <- min(object$persistence["alpha"]-parameters["beta"],adamCoefBounds["beta",2]);
                }
            }
            # Check, if there are gammas
            if(any(substr(parametersNames,1,5)=="gamma")){
                gammas <- which(substr(parametersNames,1,5)=="gamma");
                adamCoefBounds[gammas,1] <- apply(cbind(adamCoefBounds[gammas,1],-parameters[gammas]),1,max);
                if(any(parametersNames=="alpha")){
                    adamCoefBounds[gammas,2] <- apply(cbind(adamCoefBounds[gammas,2],
                                                            (1-parameters["alpha"])-parameters[gammas]),1,min);
                }
                else{
                    adamCoefBounds[gammas,2] <- apply(cbind(adamCoefBounds[gammas,2],
                                                            (1-object$persistence["alpha"])-parameters[gammas]),1,min);
                }
            }
            # Check, if there are deltas (for xreg)
            if(any(substr(parametersNames,1,5)=="delta")){
                deltas <- which(substr(parametersNames,1,5)=="delta");
                adamCoefBounds[deltas,1] <- apply(cbind(adamCoefBounds[deltas,1],-parameters[deltas]),1,max);
                adamCoefBounds[deltas,2] <- apply(cbind(adamCoefBounds[deltas,2],1-parameters[deltas]),1,min);
            }
            # These are "usual" bounds for phi. We don't care about other bounds
            if(any(parametersNames=="phi")){
                adamCoefBounds["phi",1] <- max(-parameters["phi"],adamCoefBounds["phi",1]);
                adamCoefBounds["phi",2] <- min(1-parameters["phi"],adamCoefBounds["phi",2]);
            }
        }
        #### Admissible bounds ####
        else if(object$bounds=="admissible"){
            # Check, if there is alpha
            if(any(parametersNames=="alpha")){
                alphaBounds <- eigenBounds(object, persistence,
                                           variableNumber=which(names(object$persistence)=="alpha"));
                adamCoefBounds["alpha",1] <- max(alphaBounds[1]-parameters["alpha"],adamCoefBounds["alpha",1]);
                adamCoefBounds["alpha",2] <- min(alphaBounds[2]-parameters["alpha"],adamCoefBounds["alpha",2]);
            }
            # Check, if there is beta
            if(any(parametersNames=="beta")){
                betaBounds <- eigenBounds(object, persistence,
                                          variableNumber=which(names(object$persistence)=="beta"));
                adamCoefBounds["beta",1] <- max(betaBounds[1]-parameters["beta"],adamCoefBounds["beta",1]);
                adamCoefBounds["beta",2] <- min(betaBounds[2]-parameters["beta"],adamCoefBounds["beta",2]);
            }
            # Check, if there are gammas
            if(any(substr(parametersNames,1,5)=="gamma")){
                gammas <- which(substr(parametersNames,1,5)=="gamma");
                for(i in 1:length(gammas)){
                    gammaBounds <- eigenBounds(object, persistence,
                                               variableNumber=which(substr(names(object$persistence),1,5)=="gamma")[i]);
                    adamCoefBounds[gammas[i],1] <- max(gammaBounds[1]-parameters[gammas[i]],adamCoefBounds[gammas[i],1]);
                    adamCoefBounds[gammas[i],2] <- min(gammaBounds[2]-parameters[gammas[i]],adamCoefBounds[gammas[i],2]);
                }
            }
            # Check, if there are deltas (for xreg)
            if(any(substr(parametersNames,1,5)=="delta")){
                deltas <- which(substr(parametersNames,1,5)=="delta");
                for(i in 1:length(deltas)){
                    deltaBounds <- eigenBounds(object, persistence,
                                               variableNumber=which(substr(names(object$persistence),1,5)=="delta")[i]);
                    adamCoefBounds[deltas[i],1] <- max(deltaBounds[1]-parameters[deltas[i]],adamCoefBounds[deltas[i],1]);
                    adamCoefBounds[deltas[i],2] <- min(deltaBounds[2]-parameters[deltas[i]],adamCoefBounds[deltas[i],2]);
                }
            }
        }

        # Restrictions on the initials for the multiplicative models (greater than zero)
        # Level
        # if(errorType(object)=="M" && any(parametersNames=="level")){
        #     adamCoefBounds["level",1] <- max(-parameters["level"],adamCoefBounds["level",1]);
        #     adamCoefBounds["level",2] <- max(-parameters["level"],adamCoefBounds["level",2]);
        # }
        adamModelType <- modelType(object);
        # Trend
        if(substr(adamModelType,2,2)=="M" && any(parametersNames=="trend")){
            adamCoefBounds["trend",1] <- max(-parameters["trend"],adamCoefBounds["trend",1]);
            adamCoefBounds["trend",2] <- max(-parameters["trend"],adamCoefBounds["trend",2]);
        }
        # Seasonality
        if(substr(adamModelType,nchar(adamModelType),nchar(adamModelType))=="M" &&
           any(substr(parametersNames,1,8)=="seasonal")){
            seasonals <- which(substr(parametersNames,1,8)=="seasonal");
            adamCoefBounds[seasonals,1] <- max(-parameters[seasonals],adamCoefBounds[seasonals,1]);
            adamCoefBounds[seasonals,2] <- max(-parameters[seasonals],adamCoefBounds[seasonals,2]);
        }
    }

    # Correct the bounds for the ARIMA model
    if(arimaModel){
        #### Deal with ARIMA parameters ####
        ariPolynomial <- object$other$polynomial$ariPolynomial;
        arPolynomial <- object$other$polynomial$arPolynomial;
        maPolynomial <- object$other$polynomial$maPolynomial;
        nonZeroARI <- object$other$ARIMAIndices$nonZeroARI;
        nonZeroMA <- object$other$ARIMAIndices$nonZeroMA;
        arPolynomialMatrix <- object$other$arPolynomialMatrix;
        # Locate all thetas for ARIMA
        thetas <- which(substr(parametersNames,1,5)=="theta");
        # Locate phi for ARIMA (they are always phi1, phi2 etc)
        phis <- which((substr(parametersNames,1,3)=="phi") & (nchar(parametersNames)>3));
        # Do loop for thetas
        if(length(thetas)>0){
            # MA parameters
            for(i in 1:length(thetas)){
                # In this case, we check, where the standard condition is violated for an element of persistence,
                # and then substitute the ARI part from that.
                psiBounds <- eigenBounds(object, persistence,
                                           variableNumber=which(substr(names(object$persistence),1,3)=="psi")[nonZeroMA[i,2]]);
                # If there are ARI elements in persistence, subtract (-(-x)) them to get proper bounds
                if(any(nonZeroARI[,2]==i)){
                    ariIndex <- which(nonZeroARI[,2]==i);
                    adamCoefBounds[thetas[i],1] <- max(psiBounds[1]-parameters[thetas[i]]+ariPolynomial[nonZeroARI[ariIndex,1]],
                                                       adamCoefBounds[thetas[i],1]);
                    adamCoefBounds[thetas[i],2] <- min(psiBounds[2]-parameters[thetas[i]]+ariPolynomial[nonZeroARI[ariIndex,1]],
                                                       adamCoefBounds[thetas[i],2]);
                }
                else{
                    adamCoefBounds[thetas[i],1] <- max(psiBounds[1]-parameters[thetas[i]], adamCoefBounds[thetas[i],1]);
                    adamCoefBounds[thetas[i],2] <- min(psiBounds[2]-parameters[thetas[i]], adamCoefBounds[thetas[i],2]);
                }
            }
        }
        # Locate phi for ARIMA (they are always phi1, phi2 etc)
        if(length(phis)>0){
            # AR parameters
            for(i in 1:length(phis)){
                # Get bounds for AR based on stationarity condition
                phiBounds <- arPolinomialsBounds(arPolynomialMatrix, arPolynomial,
                                                 which(arPolynomial==arPolynomial[arPolynomial!=0][-1][i]));

                adamCoefBounds[phis[i],1] <- max(phiBounds[1]-parameters[phis[i]], adamCoefBounds[phis[i],1]);
                adamCoefBounds[phis[i],2] <- min(phiBounds[2]-parameters[phis[i]], adamCoefBounds[phis[i],2]);
            }
        }
    }

    adamReturn <- cbind(adamSD,adamCoefBounds);
    colnames(adamReturn) <- c("S.E.",
                              paste0((1-level)/2*100,"%"), paste0((1+level)/2*100,"%"));

    # If parm was not provided, return everything.
    if(!exists("parm",inherits=FALSE)){
        parm <- names(adamSD);
    }

    return(adamReturn[parm,,drop=FALSE]);
}

#' @export
coef.adam <- function(object, ...){
    return(object$B);
}


#' @importFrom stats sigma
#' @export
sigma.adam <- function(object, ...){
    df <- (nobs(object, all=FALSE)-nparam(object));
    # If the sample is too small, then use biased estimator
    if(df<=0){
        df[] <- nobs(object);
    }
    return(sqrt(switch(object$distribution,
                       "dnorm"=,
                       "dlaplace"=,
                       "ds"=,
                       "dgnorm"=,
                       "dt"=,
                       "dlogis"=,
                       "dalaplace"=sum(residuals(object)^2),
                       "dlnorm"=,
                       "dllaplace"=,
                       "dls"=sum(log(residuals(object))^2),
                       "dlgnorm"=sum(log(residuals(object)-object$scale^2/2)^2),
                       "dinvgauss"=sum((residuals(object)-1)^2))
                /df));
}

#' @export
summary.adam <- function(object, level=0.95, ...){
    ourReturn <- list(model=object$model,responseName=all.vars(formula(object))[1]);

    occurrence <- NULL;
    if(is.occurrence(object$occurrence)){
        occurrence <- switch(object$occurrence$occurrence,
                             "f"=,
                             "fixed"="Fixed probability",
                             "o"=,
                             "odds-ratio"="Odds ratio",
                             "i"=,
                             "inverse-odds-ratio"="Inverse odds ratio",
                             "d"=,
                             "direct"="Direct",
                             "g"=,
                             "general"="General");
    }
    ourReturn$occurrence <- occurrence;
    ourReturn$distribution <- object$distribution;

    # Collect parameters and their standard errors
    parametersValues <- coef(object);
    if(!is.null(parametersValues)){
        parametersConfint <- confint(object, level=level, ...);
        if(is.null(parametersValues)){
            if(ncol(object$data)>1 && all(object$persistenceXreg!=0)){
                parametersValues <- c(object$persistence,object$persistenceXreg,object$initial,object$initialXreg);
            }
            else{
                parametersValues <- c(object$persistence,object$initial);
            }
            warning(paste0("Parameters are not available. You have probably provided them in the model, ",
                           "so there was nothing to estimate. We extracted smoothing parameters and initials."),
                    call.=FALSE);
        }
        parametersConfint[,2:3] <- parametersValues + parametersConfint[,2:3];
        parametersTable <- cbind(parametersValues,parametersConfint);
        rownames(parametersTable) <- rownames(parametersConfint);
        colnames(parametersTable) <- c("Estimate","Std. Error",
                                       paste0("Lower ",(1-level)/2*100,"%"),
                                       paste0("Upper ",(1+level)/2*100,"%"));
        ourReturn$coefficients <- parametersTable;
    }
    ourReturn$loss <- object$loss;
    ourReturn$lossValue <- object$lossValue;
    ourReturn$nobs <- nobs(object);
    ourReturn$nparam <- nparam(object);
    ourReturn$nParam <- object$nParam;
    ourReturn$call <- object$call;
    ourReturn$other <- object$other;

    if(object$loss=="likelihood" ||
       (any(object$loss==c("MSE","MSEh","MSCE")) & (object$distribution=="dnorm")) ||
       (any(object$loss==c("MAE","MAEh","MACE")) & (object$distribution=="dlaplace")) ||
       (any(object$loss==c("HAM","HAMh","CHAM")) & (object$distribution=="ds"))){
        ICs <- c(AIC(object),AICc(object),BIC(object),BICc(object));
        names(ICs) <- c("AIC","AICc","BIC","BICc");
        ourReturn$ICs <- ICs;
    }
    return(structure(ourReturn, class="summary.adam"));
}

#' @export
as.data.frame.summary.adam <- function(x, ...){
    return(as.data.frame(x$coefficients, ...));
}

#' @export
summary.adamCombined <- function(object, ...){
    return(print.adamCombined(object, ...));
}

#' @export
print.summary.adam <- function(x, ...){
    ellipsis <- list(...);
    if(!any(names(ellipsis)=="digits")){
        digits <- 4;
    }
    else{
        digits <- ellipsis$digits;
    }

    cat(paste0("Model estimated using ",x$call[[1]],"() function: ",x$model));
    cat("\nResponse variable:", paste0(x$responseName,collapse=""));

    if(!is.null(x$occurrence)){
        cat("\nOccurrence model type:",x$occurrence);
    }

    distrib <- switch(x$distribution,
                      "dnorm" = "Normal",
                      "dlaplace" = "Laplace",
                      "ds" = "S",
                      "dgnorm" = paste0("Generalised Normal with beta=",round(x$other$beta,digits)),
                      "dlogis" = "Logistic",
                      "dt" = paste0("Student t with nu=",round(x$other$nu, digits)),
                      "dalaplace" = paste0("Asymmetric Laplace with alpha=",round(x$other$alpha,digits)),
                      "dlnorm" = "Log Normal",
                      "dllaplace" = "Log Laplace",
                      "dls" = "Log S",
                      "dlgnorm" = paste0("Log Generalised Normal with beta=",round(x$other$beta,digits)),
                      # "dbcnorm" = paste0("Box-Cox Normal with lambda=",round(x$other$lambda,2)),
                      "dinvgauss" = "Inverse Gaussian"
    );
    if(!is.null(x$occurrence)){
        distrib <- paste0("\nMixture of Bernoulli and ", distrib);
    }
    cat("\nDistribution used in the estimation:", distrib);

    cat("\nLoss function type:",x$loss);
    if(!is.null(x$lossValue)){
        cat("; Loss function value:",round(x$lossValue,digits));
        if(any(x$loss==c("LASSO","RIDGE"))){
            cat("; lambda=",x$other$lambda);
        }
    }

    if(!is.null(x$coefficients)){
        cat("\nCoefficients:\n");
        print(round(x$coefficients,digits));
    }
    else{
        cat("\nAll coefficients were provided");
    }

    cat("\nSample size:", x$nobs);
    cat("\nNumber of estimated parameters:", x$nparam);
    cat("\nNumber of degrees of freedom:", x$nobs-x$nparam);
    if(x$nParam[2,4]>0){
        cat("\nNumber of provided parameters:", x$nParam[2,4]);
    }

    if(x$loss=="likelihood" ||
       (any(x$loss==c("MSE","MSEh","MSCE")) & (x$distribution=="dnorm")) ||
       (any(x$loss==c("MAE","MAEh","MACE")) & (x$distribution=="dlaplace")) ||
       (any(x$loss==c("HAM","HAMh","CHAM")) & (x$distribution=="ds"))){
        cat("\nInformation criteria:\n");
        print(round(x$ICs,digits));
    }
    else{
        cat("\nInformation criteria are unavailable for the chosen loss & distribution.\n");
    }
}

#' @export
vcov.adam <- function(object, ...){
    ellipsis <- list(...);
    # If the forecast is in numbers, then use its length as a horizon
    if(any(!is.na(object$forecast))){
        h <- length(object$forecast)
    }
    else{
        h <- 0;
    }
    modelReturn <- suppressWarnings(adam(object$data, h=0, model=object, formula=formula(object),
                                         FI=TRUE, stepSize=ellipsis$stepSize));
    # If any row contains all zeroes, then it means that the variable does not impact the likelihood. Invert the matrix without it.
    brokenVariables <- apply(modelReturn$FI==0,1,all);
    # If there are issues, try the same stuff, but with a different step size for hessian
    if(any(brokenVariables)){
        modelReturn <- suppressWarnings(adam(object$data, h=0, model=object, formula=formula(object),
                                             FI=TRUE, stepSize=.Machine$double.eps^(1/6)));
        brokenVariables <- apply(modelReturn$FI==0,1,all);
    }
    if(any(eigen(modelReturn$FI,only.values=TRUE)$values<0)){
        warning(paste0("Observed Fisher Information is not positive semi-definite, ",
                       "which means that the likelihood was not maximised properly. ",
                       "Consider reestimating the model, tuning the optimiser."), call.=FALSE);
    }
    FIMatrix <- modelReturn$FI[!brokenVariables,!brokenVariables,drop=FALSE];

    vcovMatrix <- try(chol2inv(chol(FIMatrix)), silent=TRUE);
    if(inherits(vcovMatrix,"try-error")){
        vcovMatrix <- try(solve(FIMatrix, diag(ncol(FIMatrix)), tol=1e-20), silent=TRUE);
        if(inherits(vcovMatrix,"try-error")){
            warning(paste0("Sorry, but the hessian is singular, so we could not invert it.\n",
                           "We failed to produce the covariance matrix of parameters."),
                    call.=FALSE);
            vcovMatrix <- diag(1e+100,ncol(FIMatrix));
        }
    }
    # If there were broken variables, reproduce the zero elements.
    # Reuse FI object in order to preserve memory. The names of cols / rows should be fine.
    modelReturn$FI[!brokenVariables,!brokenVariables] <- vcovMatrix;
    modelReturn$FI[brokenVariables,] <- modelReturn$FI[,brokenVariables] <- Inf;

    # Just in case, take absolute values for the diagonal (in order to avoid possible issues with FI)
    diag(modelReturn$FI) <- abs(diag(modelReturn$FI));

    return(modelReturn$FI);
}

#### Residuals and actuals functions ####

#' @importFrom greybox actuals
#' @export
actuals.adam <- function(object, all=TRUE, ...){
    responseName <- all.vars(formula(object))[1];
    if(all){
        return(object$data[,responseName]);
    }
    else{
        return(object$y[object$y!=0,responseName]);
    }
}

#' @export
nobs.adam <- function(object, ...){
    return(length(actuals(object, ...)));
}

#' @export
residuals.adam <- function(object, ...){
    return(switch(object$distribution,
                  "dlnorm"=,
                  "dllaplace"=,
                  "dls"=,
                  "dlgnorm"=,
                  "dinvgauss"=switch(errorType(object),
                                     # abs() is needed in order to avoid the weird cases
                                     "A"=abs(1+object$residuals/fitted(object)),
                                     "M"=1+object$residuals),
                  "dnorm"=,
                  "dlaplace"=,
                  "ds"=,
                  "dgnorm"=,
                  "dlogis"=,
                  "dt"=,
                  "dalaplace"=,
                  object$residuals));
}

#' Multiple steps ahead forecast errors
#'
#' The function extracts 1 to h steps ahead forecast errors from the model.
#'
#' The errors correspond to the error term epsilon_t in the ETS models. Don't forget
#' that different models make different assumptions about epsilon_t and / or 1+epsilon_t.
#'
#' @template ssAuthor
#' @template ssKeywords
#'
#' @param object Model estimated using one of the forecasting functions.
#' @param h The forecasting horizon to use.
#' @param ... Currently nothing is accepted via ellipsis.
#' @return The matrix with observations in rows and h steps ahead values in columns.
#' So, the first row corresponds to the forecast produced from the 0th observation
#' from 1 to h steps ahead.
#' @seealso \link[stats]{residuals},
#' @examples
#'
#' x <- rnorm(100,0,1)
#' ourModel <- adam(x)
#' rmultistep(ourModel, h=13)
#'
#' @export rmultistep
rmultistep <- function(object, h=10, ...) UseMethod("rmultistep")

#' @export
rmultistep.default <- function(object, h=10, ...){
    return(NULL);
}

#' @export
rmultistep.adam <- function(object, h=10, ...){
    yClasses <- class(actuals(object));

    # Model type
    model <- modelType(object);
    Etype <- errorType(object);
    Ttype <- substr(model,2,2);
    Stype <- substr(model,nchar(model),nchar(model));

    # Technical parameters
    lagsModelAll <- modelLags(object);
    lagsModelMax <- max(lagsModelAll);
    lagsOriginal <- lags(object);
    if(Ttype!="N"){
        lagsOriginal <- c(1,lagsOriginal);
    }
    if(!is.null(object$initial$seasonal)){
        if(is.list(object$initial$seasonal)){
            componentsNumberETSSeasonal <- length(object$initial$seasonal);
        }
        else{
            componentsNumberETSSeasonal <- 1;
        }
    }
    else{
        componentsNumberETSSeasonal <- 0;
    }
    componentsNumberETS <- length(object$initial$level) + length(object$initial$trend) + componentsNumberETSSeasonal;
    componentsNumberARIMA <- sum(substr(colnames(object$states),1,10)=="ARIMAState");
    if(ncol(object$data)>1){
        xregNumber <- ncol(object$data)-1;
    }
    else{
        xregNumber <- 0;
    }
    obsInSample <- nobs(object);

    # Function returns the matrix with multi-step errors
    if(is.occurrence(object$occurrence)){
        ot <- matrix(actuals(object$occurrence),obsInSample,1);
    }
    else{
        ot <- matrix(1,obsInSample,1);
    }
    adamProfiles <- adamProfileCreator(lagsModelAll, max(lagsModelAll), obsInSample,
                                       lagsOriginal, time(actuals(object)), yClasses);
    profilesRecentTable <- adamProfiles$recent;
    profilesObservedTable <- adamProfiles$observed;

    # Return multi-step errors matrix
    if(any(yClasses=="ts")){
        return(ts(adamErrorerWrap(t(object$states), object$measurement, object$transition,
                                  lagsModelAll, profilesObservedTable, profilesRecentTable,
                                  Etype, Ttype, Stype,
                                  componentsNumberETS, componentsNumberETSSeasonal,
                                  componentsNumberARIMA, xregNumber, h,
                                  matrix(actuals(object),obsInSample,1), ot),
                  start=start(actuals(object)), frequency=frequency(actuals(object))));
    }
    else{
        return(zoo(adamErrorerWrap(t(object$states), object$measurement, object$transition,
                                   lagsModelAll, profilesObservedTable, profilesRecentTable,
                                   Etype, Ttype, Stype,
                                   componentsNumberETS, componentsNumberETSSeasonal,
                                   componentsNumberARIMA, xregNumber, h,
                                   matrix(actuals(object),obsInSample,1), ot),
                  order.by=time(actuals(object))));
    }
}

#' @importFrom stats rstandard
#' @export
rstandard.adam <- function(model, ...){
    obs <- nobs(model);
    df <- obs - nparam(model);
    errors <- residuals(model);
    # If this is an occurrence model, then only modify the non-zero obs
    # Also, if there are NAs in actuals, consider them as occurrence
    if(is.occurrence(model$occurrence)){
        residsToGo <- which(actuals(model$occurrence)!=0 & !is.na(actuals(model)));
    }
    else{
        residsToGo <- c(1:obs);
    }

    if(any(model$distribution==c("dt","dnorm"))){
        return((errors - mean(errors[residsToGo])) / sqrt(model$scale^2 * obs / df));
    }
    else if(model$distribution=="ds"){
        return((errors - mean(errors[residsToGo])) / (model$scale * obs / df)^2);
    }
    else if(model$distribution=="dls"){
        errors[] <- log(errors);
        return(exp((errors - mean(errors[residsToGo])) / (model$scale * obs / df)^2));
    }
    else if(model$distribution=="dgnorm"){
        return((errors - mean(errors[residsToGo])) / (model$scale^model$other$beta * obs / df)^{1/model$other$beta});
    }
    else if(model$distribution=="dlgnorm"){
        errors[] <- log(errors);
        return(exp((errors - mean(errors[residsToGo])) / (model$scale^model$other$beta * obs / df)^{1/model$other$beta}));
    }
    else if(model$distribution=="dinvgauss"){
        return(errors / mean(errors[residsToGo]));
    }
    else if(model$distribution=="dlnorm"){
        # Debias the residuals
        errors[] <- log(errors) - model$scale^2/2;
        return(exp((errors - mean(errors[residsToGo])) / sqrt(model$scale^2 * obs / df)));
    }
    else if(model$distribution=="dllaplace"){
        errors[] <- log(errors);
        return(exp((errors - mean(errors[residsToGo])) / model$scale * obs / df));
    }
    else{
        return(errors / model$scale * obs / df);
    }
}

#' @importFrom stats rstudent
#' @export
rstudent.adam <- function(model, ...){
    obs <- nobs(model);
    df <- obs - nparam(model) - 1;
    rstudentised <- errors <- residuals(model);
    # If this is an occurrence model, then only modify the non-zero obs
    # Also, if there are NAs in actuals, consider them as occurrence
    if(is.occurrence(model$occurrence)){
        residsToGo <- which(actuals(model$occurrence)!=0 & !is.na(actuals(model)));
    }
    else{
        residsToGo <- c(1:obs);
    }
    if(any(model$distribution==c("dt","dnorm"))){
        errors[] <- errors - mean(errors);
        for(i in residsToGo){
            rstudentised[i] <- errors[i] / sqrt(sum(errors[-i]^2,na.rm=TRUE) / df);
        }
    }
    else if(model$distribution=="dlaplace"){
        errors[] <- errors - mean(errors);
        for(i in residsToGo){
            rstudentised[i] <- errors[i] / (sum(abs(errors[-i]),na.rm=TRUE) / df);
        }
    }
    else if(model$distribution=="dlnorm"){
        errors[] <- log(errors) - mean(log(errors)) - model$scale^2/2;
        for(i in residsToGo){
            rstudentised[i] <- exp(errors[i] / sqrt(sum(errors[-i]^2,na.rm=TRUE) / df));
        }
    }
    else if(model$distribution=="dllaplace"){
        errors[] <- log(errors) - mean(log(errors));
        for(i in residsToGo){
            rstudentised[i] <- exp(errors[i] / (sum(abs(errors[-i]),na.rm=TRUE) / df));
        }
    }
    else if(model$distribution=="ds"){
        errors[] <- errors - mean(errors);
        for(i in residsToGo){
            rstudentised[i] <- errors[i] / (sum(sqrt(abs(errors[-i])),na.rm=TRUE) / (2*df))^2;
        }
    }
    else if(model$distribution=="dls"){
        errors[] <- log(errors) - mean(log(errors));
        for(i in residsToGo){
            rstudentised[i] <- exp(errors[i] / (sum(sqrt(abs(errors[-i])),na.rm=TRUE) / (2*df))^2);
        }
    }
    else if(model$distribution=="dgnorm"){
        errors[] <- errors - mean(errors);
        for(i in residsToGo){
            rstudentised[i] <- errors[i] /  (sum(abs(errors[-i])^model$other$beta) * (model$other$beta/df))^{1/model$other$beta};
        }
    }
    else if(model$distribution=="dlgnorm"){
        errors[] <- log(errors) - mean(log(errors));
        for(i in residsToGo){
            rstudentised[i] <- errors[i] /  (sum(abs(errors[-i])^model$other$beta) * (model$other$beta/df))^{1/model$other$beta};
        }
    }
    else if(model$distribution=="dalaplace"){
        for(i in residsToGo){
            rstudentised[i] <- errors[i] / (sum(errors[-i] * (model$other$alpha - (errors[-i]<=0)*1),na.rm=TRUE) / df);
        }
    }
    else if(model$distribution=="dlogis"){
        errors[] <- errors - mean(errors);
        for(i in residsToGo){
            rstudentised[i] <- errors[i] / (sqrt(sum(errors[-i]^2,na.rm=TRUE) / df) * sqrt(3) / pi);
        }
    }
    else if(model$distribution=="dinvgauss"){
        for(i in residsToGo){
            rstudentised[i] <- errors[i] / mean(errors[residsToGo][-i],na.rm=TRUE);
        }
    }
    else{
        for(i in residsToGo){
            rstudentised[i] <- errors[i] / sqrt(sum(errors[-i]^2,na.rm=TRUE) / df);
        }
    }
    return(rstudentised);
}

#' @importFrom greybox outlierdummy
#' @export
outlierdummy.adam <- function(object, level=0.999, type=c("rstandard","rstudent"), ...){
    # Function returns the matrix of dummies with outliers
    type <- match.arg(type);
    errors <- switch(type,"rstandard"=rstandard(object),"rstudent"=rstudent(object));
    statistic <- switch(object$distribution,
                      "dlaplace"=,
                      "dllaplace"=qlaplace(c((1-level)/2, (1+level)/2), 0, 1),
                      "dalaplace"=qalaplace(c((1-level)/2, (1+level)/2), 0, 1, object$other$alpha),
                      "dlogis"=qlogis(c((1-level)/2, (1+level)/2), 0, 1),
                      "dt"=qt(c((1-level)/2, (1+level)/2), nobs(object)-nparam(object)),
                      "dgnorm"=,
                      "dlgnorm"=qgnorm(c((1-level)/2, (1+level)/2), 0, 1, object$other$beta),
                      "ds"=,
                      "dls"=qs(c((1-level)/2, (1+level)/2), 0, 1),
                      # In the next one, the scale is debiased, taking n-k into account
                      "dinvgauss"=qinvgauss(c((1-level)/2, (1+level)/2), mean=1,
                                            dispersion=object$scale * nobs(object) /
                                                (nobs(object)-nparam(object))),
                      qnorm(c((1-level)/2, (1+level)/2), 0, 1));
    if(any(object$distribution==c("dlnorm","dllaplace","dls","dlgnorm"))){
        errors[] <- log(errors);
    }
    outliersID <- which(errors>statistic[2] | errors<statistic[1]);
    outliersNumber <- length(outliersID);
    if(outliersNumber>0){
        outliers <- matrix(0, nobs(object), outliersNumber,
                           dimnames=list(rownames(object$data),
                                         paste0("outlier",c(1:outliersNumber))));
        outliers[cbind(outliersID,c(1:outliersNumber))] <- 1;
    }
    else{
        outliers <- NULL;
    }

    return(structure(list(outliers=outliers, statistic=statistic, id=outliersID,
                          level=level, type=type),
                     class="outlierdummy"));
}

#### Predict and forecast functions ####
#' @export
predict.adam <- function(object, newdata=NULL, interval=c("none", "confidence", "prediction"),
                         level=0.95, side=c("both","upper","lower"), ...){

    interval <- match.arg(interval);
    obsInSample <- nobs(object);

    # Indices and classes of the original data
    yIndex <- time(actuals(object));
    yClasses <- class(actuals(object));
    if(any(yClasses=="ts")){
        # ts structure
        yStart <- yIndex[1];
        yFrequency <- frequency(actuals(object));
    }

    # Check if newdata is provided
    if(!is.null(newdata)){
        # If this is not a matrix / data.frame, then convert to one
        if(!is.data.frame(newdata) && !is.matrix(newdata)){
            newdata <- as.data.frame(newdata);
            colnames(newdata) <- "xreg";
        }
        h <- nrow(newdata);
        # If the newdata is provided, then just do forecasts for that part
        if(any(interval==c("none","prediction","confidence"))){
            if(interval==c("prediction")){
                interval[] <- "simulated";
            }
            return(forecast(object, h=h, newdata=newdata,
                            interval=interval,
                            level=level, side=side, ...));
        }
    }
    else{
        # If there are no newdata, then we need to produce fitted with / without interval
        if(interval=="none"){
            return(structure(list(mean=fitted(object), lower=NA, upper=NA, model=object,
                                  level=level, interval=interval, side=side),
                             class=c("adam.predict","adam.forecast")));
        }
        # Otherwise we do one-step-ahead prediction / confidence interval
        else{
            yForecast <- fitted(object);
        }
    }

    ##### Prediction interval for in sample only! #####
    side <- match.arg(side);

    if(length(level)>1){
        warning(paste0("Sorry, but we only support scalar for the level, ",
                       "when constructing in-sample interval. ",
                       "Using the first provided value."),
                call.=FALSE);
        level <- level[1];
    }
    # Fix just in case a silly user used 95 etc instead of 0.95
    if(level>1){
        level[] <- level / 100;
    }

    # Basic parameters
    model <- modelType(object);
    Etype <- errorType(object);

    # Extract variance and amend it in case of confidence interval
    s2 <- sigma(object)^2;

    # If this is a mixture model, produce forecasts for the occurrence
    if(!is.null(object$occurrence)){
        occurrenceModel <- TRUE;
        pForecast <- fitted(object$occurrence);
    }
    else{
        occurrenceModel <- FALSE;
        pForecast <- rep(1, obsInSample);
    }

    # If this is an occurrence model, then take probability into account in the level.
    if(occurrenceModel && (interval=="prediction")){
        levelNew <- (level-(1-pForecast))/pForecast;
        levelNew[levelNew<0] <- 0;
    }
    else{
        levelNew <- level;
    }

    levelLow <- levelUp <- vector("numeric",obsInSample);
    if(side=="both"){
        levelLow[] <- (1-levelNew)/2;
        levelUp[] <- (1+levelNew)/2;
    }
    else if(side=="upper"){
        levelLow[] <- rep(0,length(levelNew));
        levelUp[] <- levelNew;
    }
    else{
        levelLow[] <- 1-levelNew;
        levelUp[] <- rep(1,length(levelNew));
    }
    levelLow[levelLow<0] <- 0;
    levelUp[levelUp<0] <- 0;

    nLevels <- 1;
    # Matrices for levels

    # Create necessary matrices for the forecasts
    if(any(yClasses=="ts")){
        yUpper <- yLower <- ts(matrix(0,obsInSample,nLevels), start=yStart, frequency=yFrequency);
    }
    else{
        yUpper <- yLower <- zoo(matrix(0,obsInSample,nLevels), order.by=yIndex);
    }
    colnames(yLower) <- switch(side,
                               "both"=paste0("Lower bound (",(1-level)/2*100,"%)"),
                               "lower"=paste0("Lower bound (",(1-level)*100,"%)"),
                               "upper"=rep("Lower 0%",nLevels));

    colnames(yUpper) <- switch(side,
                               "both"=paste0("Upper bound (",(1+level)/2*100,"%)"),
                               "lower"=rep("Upper 100%",nLevels),
                               "upper"=paste0("Upper bound (",level*100,"%)"));

    #### Call refit if this is confidence ####
    if(interval=="confidence"){
        yFittedMatrix <- refit(object, ...);
        for(i in 1:obsInSample){
            yUpper[i] <- quantile(yFittedMatrix$refitted[i,], levelLow[i], na.rm=TRUE);
            yLower[i] <- quantile(yFittedMatrix$refitted[i,], levelUp[i], na.rm=TRUE);
        }
        return(structure(list(mean=yForecast, lower=yLower, upper=yUpper, model=object,
                              level=level, interval=interval, side=side),
                         class=c("adam.predict","adam.forecast")))
    }

    #### Produce the prediction intervals ####
    if(object$distribution=="dnorm"){
        if(Etype=="A"){
            yLower[] <- qnorm(levelLow, 0, sqrt(s2));
            yUpper[] <- qnorm(levelUp, 0, sqrt(s2));
        }
        else{
            yLower[] <- qnorm(levelLow, 1, sqrt(s2));
            yUpper[] <- qnorm(levelUp, 1, sqrt(s2));
        }
    }
    else if(object$distribution=="dlaplace"){
        if(Etype=="A"){
            yLower[] <- qlaplace(levelLow, 0, sqrt(s2/2));
            yUpper[] <- qlaplace(levelUp, 0, sqrt(s2/2));
        }
        else{
            yLower[] <- qlaplace(levelLow, 1, sqrt(s2/2));
            yUpper[] <- qlaplace(levelUp, 1, sqrt(s2/2));
        }
    }
    else if(object$distribution=="ds"){
        if(Etype=="A"){
            yLower[] <- qs(levelLow, 0, (s2/120)^0.25);
            yUpper[] <- qs(levelUp, 0, (s2/120)^0.25);
        }
        else{
            yLower[] <- qs(levelLow, 1, (s2/120)^0.25);
            yUpper[] <- qs(levelUp, 1, (s2/120)^0.25);
        }
    }
    else if(object$distribution=="dgnorm"){
        alpha <- sqrt(s2*(gamma(1/object$other$beta)/gamma(3/object$other$beta)));
        if(Etype=="A"){
            yLower[] <- suppressWarnings(qgnorm(levelLow, 0, alpha, object$other$beta));
            yUpper[] <- suppressWarnings(qgnorm(levelUp, 0, alpha, object$other$beta));
        }
        else{
            yLower[] <- suppressWarnings(qgnorm(levelLow, 1, alpha, object$other$beta));
            yUpper[] <- suppressWarnings(qgnorm(levelUp, 1, alpha, object$other$beta));
        }
    }
    else if(object$distribution=="dlogis"){
        if(Etype=="A"){
            yLower[] <- qlogis(levelLow, 0, sqrt(s2*3)/pi);
            yUpper[] <- qlogis(levelUp, 0, sqrt(s2*3)/pi);
        }
        else{
            yLower[] <- qlogis(levelLow, 1, sqrt(s2*3)/pi);
            yUpper[] <- qlogis(levelUp, 1, sqrt(s2*3)/pi);
        }
    }
    else if(object$distribution=="dt"){
        df <- nobs(object) - nparam(object);
        if(Etype=="A"){
            yLower[] <- sqrt(s2)*qt(levelLow, df);
            yUpper[] <- sqrt(s2)*qt(levelUp, df);
        }
        else{
            yLower[] <- (1 + sqrt(s2)*qt(levelLow, df));
            yUpper[] <- (1 + sqrt(s2)*qt(levelUp, df));
        }
    }
    else if(object$distribution=="dalaplace"){
        alpha <- object$other$beta;
        if(Etype=="A"){
            yLower[] <- qalaplace(levelLow, 0,
                                  sqrt(s2*alpha^2*(1-alpha)^2/(alpha^2+(1-alpha)^2)), alpha);
            yUpper[] <- qalaplace(levelUp, 0,
                                  sqrt(s2*alpha^2*(1-alpha)^2/(alpha^2+(1-alpha)^2)), alpha);
        }
        else{
            yLower[] <- qalaplace(levelLow, 1,
                                  sqrt(s2*alpha^2*(1-alpha)^2/(alpha^2+(1-alpha)^2)), alpha);
            yUpper[] <- qalaplace(levelUp, 1,
                                  sqrt(s2*alpha^2*(1-alpha)^2/(alpha^2+(1-alpha)^2)), alpha);
        }
    }
    else if(object$distribution=="dlnorm"){
        # Take into account the logN restrictions
        yLower[] <- qlnorm(levelLow, -(1-sqrt(abs(1-s2)))^2, sqrt(2*(1-sqrt(abs(1-s2)))));
        yUpper[] <- qlnorm(levelUp, -(1-sqrt(abs(1-s2)))^2, sqrt(2*(1-sqrt(abs(1-s2)))));
    }
    else if(object$distribution=="dllaplace"){
        yLower[] <- exp(qlaplace(levelLow, 0, sqrt(s2/2)));
        yUpper[] <- exp(qlaplace(levelUp, 0, sqrt(s2/2)));
    }
    else if(object$distribution=="dls"){
        yLower[] <- exp(qs(levelLow, 0, (s2/120)^0.25));
        yUpper[] <- exp(qs(levelUp, 0, (s2/120)^0.25));
    }
    else if(object$distribution=="dlgnorm"){
        alpha <- sqrt(s2*(gamma(1/object$other$beta)/gamma(3/object$other$beta)));
        yLower[] <- suppressWarnings(exp(qgnorm(levelLow, 0, alpha, object$other$beta)));
        yUpper[] <- suppressWarnings(exp(qgnorm(levelUp, 0, alpha, object$other$beta)));
    }
    else if(object$distribution=="dinvgauss"){
        yLower[] <- qinvgauss(levelLow, 1, dispersion=s2);
        yUpper[] <- qinvgauss(levelUp, 1, dispersion=s2);
    }

    #### Clean up the produced values for the interval ####
    # Make sensible values out of those weird quantiles
    if(Etype=="A"){
        yLower[levelLow==0] <- -Inf;
    }
    else{
        yLower[levelLow==0] <- 0;
    }
    yUpper[levelUp==1] <- Inf;

    # Substitute NAs and NaNs with zeroes
    if(any(is.nan(yLower)) || any(is.na(yLower))){
        yLower[is.nan(yLower)] <- 0;
        yLower[is.na(yLower)] <- 0;
    }
    if(any(is.nan(yUpper)) || any(is.na(yUpper))){
        yUpper[is.nan(yUpper)] <- 0;
        yUpper[is.na(yUpper)] <- 0;
    }

    if(Etype=="A"){
        yLower[] <- yForecast + yLower;
        yUpper[] <- yForecast + yUpper;
    }
    else{
        yLower[] <- yForecast * yLower;
        yUpper[] <- yForecast * yUpper;
    }

    return(structure(list(mean=yForecast, lower=yLower, upper=yUpper, model=object,
                          level=level, interval=interval, side=side),
                     class=c("adam.predict","adam.forecast")));
}

#' @export
plot.adam.predict <- function(x, ...){
    ellipsis <- list(...);
    if(is.null(ellipsis$ylim)){
        ellipsis$ylim <- range(c(actuals(x$model),x$mean,x$lower,x$upper),na.rm=TRUE);
    }
    ellipsis$x <- actuals(x$model);
    do.call(plot, ellipsis);
    lines(x$mean,col="purple",lwd=2,lty=2);
    if(x$interval!="none"){
        lines(x$lower,col="grey",lwd=3,lty=2);
        lines(x$upper,col="grey",lwd=3,lty=2);
    }
}

# Work in progress...
#' @param newdata The new data needed in order to produce forecasts.
#' @param nsim Number of iterations to do in case of \code{interval="simulated"}.
#' @param interval What type of mechanism to use for interval construction. The
#' recommended option is \code{interval="prediction"}, which will use analytical
#' solutions for pure additive models and simulations for the others.
#' \code{interval="simulated"} is the slowest method, but is robust to the type of
#' model. \code{interval="approximate"} (aka \code{interval="parametric"}) uses
#' analytical formulae for conditional h-steps ahead variance, but is approximate
#' for the non-additive error models. \code{interval="semiparametric"} relies on the
#' multiple steps ahead forecast error and on the assumed distribution of the error
#' term. \code{interval="nonparametric"} uses Taylor & Bunn (1999) approach with
#' quantile regressions. Finally, \code{interval="confidence"} tries to generate the
#' confidence intervals for the point forecast based on the \code{reforecast} method.
#' @param cumulative If \code{TRUE}, then the cumulative forecast and prediction
#' interval are produced instead of the normal ones. This is useful for
#' inventory control systems.
#' @param occurrence The vector containing the future occurrence variable
#' (values in [0,1]), if it is known.
#' @rdname forecast.smooth
#' @importFrom stats rnorm rlogis rt rlnorm qnorm qlogis qt qlnorm
#' @importFrom statmod rinvgauss qinvgauss
#' @importFrom greybox rlaplace rs ralaplace qlaplace qs qalaplace
#' @export
forecast.adam <- function(object, h=10, newdata=NULL, occurrence=NULL,
                          interval=c("none", "prediction", "confidence", "simulated",
                                     "approximate", "semiparametric", "nonparametric"),
                          level=0.95, side=c("both","upper","lower"), cumulative=FALSE, nsim=10000, ...){

    ellipsis <- list(...);

    interval <- match.arg(interval[1],c("none", "simulated", "approximate", "semiparametric",
                                        "nonparametric", "confidence", "parametric","prediction"));
    # If the horizon is zero, just construct fitted and potentially confidence interval thingy
    if(h<=0){
        if(all(interval!=c("none","confidence"))){
            interval[] <- "prediction";
        }
        return(predict(object, newdata=newdata,
                       interval=interval,
                       level=level, side=side, ...));
    }
    else{
        if(interval=="confidence"){
            return(reforecast(object, h=h, newdata=newdata, occurrence=occurrence,
                              interval=interval, level=level, side=side, cumulative=cumulative,
                              nsim=nsim, ...));
        }
    }

    if(interval=="parametric"){
        # warning("The parameter 'interval' does not accept 'parametric' anymore. We use 'approximate' value instead.",
        #         call.=FALSE);
        interval <- "approximate";
    }
    side <- match.arg(side);

    # Model type
    model <- modelType(object);
    Etype <- errorType(object);
    Ttype <- substr(model,2,2);
    Stype <- substr(model,nchar(model),nchar(model));

    # If this is "prediction", do simulations for multiplicative components
    if(interval=="prediction"){
        if(any(c(Etype,Ttype,Stype)=="M")){
            interval <- "simulated";
        }
        else{
            interval <- "approximate";
        }
    }

    # Technical parameters
    lagsModelAll <- modelLags(object);
    lagsModelMax <- max(lagsModelAll);
    profilesRecentTable <- object$profile;
    yClasses <- class(actuals(object));

    if(!is.null(object$initial$seasonal)){
        if(is.list(object$initial$seasonal)){
            componentsNumberETSSeasonal <- length(object$initial$seasonal);
        }
        else{
            componentsNumberETSSeasonal <- 1;
        }
    }
    else{
        componentsNumberETSSeasonal <- 0;
    }
    componentsNumberETS <- length(object$initial$level) + length(object$initial$trend) + componentsNumberETSSeasonal;
    componentsNumberARIMA <- sum(substr(colnames(object$states),1,10)=="ARIMAState");

    obsStates <- nrow(object$states);
    obsInSample <- nobs(object);

    yIndex <- time(actuals(object));
    # Create indices for the future
    if(any(yClasses=="ts")){
        yForecastIndex <- yIndex[obsInSample]+as.numeric(diff(tail(yIndex,2)))*c(1:h);
    }
    else{
        yForecastIndex <- yIndex[obsInSample]+diff(tail(yIndex,2))*c(1:h);
    }
    # Get the observed profiles
    profilesObservedTable <- adamProfileCreator(lagsModelAll, lagsModelMax, obsInSample+h,
                                                lags(object), c(yIndex,yForecastIndex),
                                                yClasses)$observed[,-c(1:obsInSample),drop=FALSE];

    if(any(yClasses=="ts")){
        # ts structure
        yForecastStart <- time(actuals(object))[obsInSample]+deltat(actuals(object));
        yFrequency <- frequency(actuals(object));
    }
    else{
        # zoo thingy
        yIndex <- time(actuals(object));
        yForecastIndex <- yIndex[obsInSample]+diff(tail(yIndex,2))*c(1:h);
    }

    # All the important matrices
    matVt <- t(object$states[obsStates-(lagsModelMax:1)+1,,drop=FALSE]);
    matWt <- tail(object$measurement,h);
    # If the forecast horizon is higher than the in-sample, duplicate the last value in matWt
    if(nrow(matWt)<h){
        matWt <- matrix(tail(matWt,1), nrow=h, ncol=ncol(matWt), dimnames=list(NULL,colnames(matWt)), byrow=TRUE);
    }
    vecG <- matrix(object$persistence, ncol=1);

    # Deal with explanatory variables
    if(ncol(object$data)>1){
        xregNumber <- length(object$initial$xreg);
        xregNames <- names(object$initial$xreg);
        if(is.null(newdata) && !is.null(object$holdout) && nrow(object$holdout)<h){
            xreg <- tail(object$data,h);
            if(!is.data.frame(newdata) & is.matrix(xreg)){
                warning("The newdata is not provided.",
                        "Predicting the explanatory variables based on what we have in-sample.",
                        call.=FALSE);
                for(i in 1:xregNumber){
                    xreg[,i] <- adam(object$data[,i+1],h=h,silent=TRUE)$forecast;
                }
            }
            else{
                warning("The newdata is not provided. Using last h in-sample observations instead.",
                        call.=FALSE);
            }
        }
        else if(is.null(newdata) && !is.null(object$holdout) && nrow(object$holdout)>=h){
            xreg <- object$holdout[1:h,,drop=FALSE];
        }
        else{
            # If this is not a matrix / data.frame, then convert to one
            if(!is.data.frame(newdata) && !is.matrix(newdata)){
                newdata <- as.data.frame(newdata);
                colnames(newdata) <- "xreg";
            }
            if(nrow(newdata)<h){
                warning(paste0("The newdata has ",nrow(newdata)," observations, while ",h," are needed. ",
                               "Using the last available values as future ones."),
                        call.=FALSE);
                newnRows <- h-nrow(newdata);
                xreg <- rbind(newdata,matrix(rep(tail(newdata,1),each=newnRows),newnRows,ncol(newdata)));
            }
            else if(nrow(newdata)>h){
                warning(paste0("The newdata has ",nrow(newdata)," observations, while only ",h," are needed. ",
                               "Using the last ",h," of them."),
                        call.=FALSE);
                xreg <- tail(newdata,h);
            }
            else{
                xreg <- newdata;
            }
        }

        # If the names are wrong, transform to data frame and expand
        if(!all(xregNames %in% colnames(xreg))){
            xreg <- as.data.frame(xreg);
        }

        # Expand the xreg if it is data frame to get the proper matrix
        if(is.data.frame(xreg)){
            testFormula <- formula(object);
            # Expand the variables. We cannot use alm, because it is based on obsInSample
            xregData <- model.frame(testFormula,data=xreg);
            # Binary, flagging factors in the data
            # Expanded stuff with all levels for factors
            if(any((attr(terms(xregData),"dataClasses")=="factor")[-1])){
                xregModelMatrix <- model.matrix(xregData,xregData,
                                                contrasts.arg=lapply(xregData[attr(terms(xregData),"dataClasses")=="factor"],
                                                                     contrasts, contrasts=FALSE));
            }
            else{
                xregModelMatrix <- model.matrix(xregData,data=xregData);
            }
            colnames(xregModelMatrix) <- make.names(colnames(xregModelMatrix), unique=TRUE);
            newdata <- as.matrix(xregModelMatrix)[,xregNames,drop=FALSE];
            rm(xregData,xregModelMatrix);
        }
        else{
            newdata <- xreg[,xregNames,drop=FALSE];
        }
        rm(xreg);

        matWt[,componentsNumberETS+componentsNumberARIMA+c(1:xregNumber)] <- newdata;
        # If this is not "adapt", then fill in the matrix with zeroes
        if(object$regressors!="adapt"){
            vecG <- matrix(c(vecG,rep(0,xregNumber)),ncol=1);
        }
    }
    else{
        xregNumber <- 0;
    }
    matF <- object$transition;

    # Produce point forecasts for non-multiplicative trend / seasonality
    if(Ttype!="M" && Stype!="M"){
        adamForecast <- adamForecasterWrap(matWt, matF,
                                           lagsModelAll, profilesObservedTable, profilesRecentTable,
                                           Etype, Ttype, Stype,
                                           componentsNumberETS, componentsNumberETSSeasonal,
                                           componentsNumberARIMA, xregNumber,
                                           h);
    }
    else{
        # If we do simulations, leave it for later
        if(interval=="simulated"){
            adamForecast <- rep(0, h);
        }
        # If we don't, do simulations to get mean
        else{
            adamForecast <- forecast(object, h=h, newdata=newdata, occurrence=occurrence,
                                     interval="simulated",
                                     level=level, side="both", cumulative=cumulative, nsim=nsim, ...)$mean;
        }
    }

    #### Make safety checks
    # If there are NaN values
    if(any(is.nan(adamForecast))){
        adamForecast[is.nan(adamForecast)] <- 0;
    }

    # If the occurrence values are provided for the holdout
    if(!is.null(occurrence) && is.numeric(occurrence)){
        pForecast <- occurrence;
    }
    else{
        # If this is a mixture model, produce forecasts for the occurrence
        if(is.occurrence(object$occurrence)){
            occurrenceModel <- TRUE;
            if(is.alm(object$occurrence)){
                pForecast <- forecast(object$occurrence,h=h,newdata=newdata)$mean;
            }
            else{
                pForecast <- forecast(object$occurrence,h=h,newdata=newdata)$mean;
            }
        }
        else{
            occurrenceModel <- FALSE;
            # If this was provided occurrence, then use provided values
            if(!is.null(object$occurrence) && !is.null(object$occurrence$occurrence) &&
               (object$occurrence$occurrence=="provided")){
                pForecast <- object$occurrence$forecast;
            }
            else{
                pForecast <- rep(1, h);
            }
        }
    }
    # Make sure that the values are of the correct length
    if(h<length(pForecast)){
        pForecast <- pForecast[1:h];
    }
    else if(h>length(pForecast)){
        pForecast <- c(pForecast, rep(tail(pForecast,1), h-length(pForecast)));
    }

    # How many levels did user asked to produce
    nLevels <- length(level);
    # Cumulative forecasts have only one observation
    if(cumulative){
        # hFinal is the number of elements we will have in the final forecast
        hFinal <- 1;
    }
    else{
        hFinal <- h;
    }

    # Create necessary matrices for the forecasts
    if(any(yClasses=="ts")){
        yForecast <- ts(vector("numeric", hFinal), start=yForecastStart, frequency=yFrequency);
        yUpper <- yLower <- ts(matrix(0,hFinal,nLevels), start=yForecastStart, frequency=yFrequency);
    }
    else{
        yForecast <- zoo(vector("numeric", hFinal), order.by=yForecastIndex);
        yUpper <- yLower <- zoo(matrix(0,hFinal,nLevels), order.by=yForecastIndex);
    }
    # Fill in the point forecasts
    if(cumulative){
        yForecast[] <- sum(adamForecast * pForecast);
    }
    else{
        yForecast[] <- adamForecast * pForecast;
    }

    if(interval!="none"){
        # Fix just in case a silly user used 95 etc instead of 0.95
        if(any(level>1)){
            level[] <- level / 100;
        }
        levelLow <- levelUp <- matrix(0,nrow=hFinal,ncol=nLevels);
        levelNew <- matrix(level,nrow=hFinal,ncol=nLevels,byrow=TRUE);

        # If this is an occurrence model, then take probability into account in the level.
        # This correction is only needed for approximate, because the others contain zeroes
        if(occurrenceModel && interval=="approximate"){
            levelNew[] <- (levelNew-(1-as.vector(pForecast)))/as.vector(pForecast);
            levelNew[levelNew<0] <- 0;
        }
        if(side=="both"){
            levelLow[] <- (1-levelNew)/2;
            levelUp[] <- (1+levelNew)/2;
        }
        else if(side=="upper"){
            levelLow[] <- 0;
            levelUp[] <- levelNew;
        }
        else{
            levelLow[] <- 1-levelNew;
            levelUp[] <- 1;
        }
        levelLow[levelLow<0] <- 0;
        levelUp[levelUp<0] <- 0;
    }

    #### Simulated interval ####
    if(interval=="simulated"){
        arrVt <- array(NA, c(componentsNumberETS+componentsNumberARIMA+xregNumber, h+lagsModelMax, nsim));
        arrVt[,1:lagsModelMax,] <- rep(matVt,nsim);
        sigmaValue <- sigma(object);
        matErrors <- matrix(switch(object$distribution,
                                   "dnorm"=rnorm(h*nsim, 0, sigmaValue),
                                   "dlaplace"=rlaplace(h*nsim, 0, sigmaValue/2),
                                   "ds"=rs(h*nsim, 0, (sigmaValue^2/120)^0.25),
                                   "dgnorm"=rgnorm(h*nsim, 0,
                                                   sigmaValue*sqrt(gamma(1/object$other$beta)/gamma(3/object$other$beta))),
                                   "dlogis"=rlogis(h*nsim, 0, sigmaValue*sqrt(3)/pi),
                                   "dt"=rt(h*nsim, obsInSample-nparam(object)),
                                   "dalaplace"=ralaplace(h*nsim, 0,
                                                         sqrt(sigmaValue^2*object$other$alpha^2*(1-object$other$alpha)^2/
                                                                  (object$other$alpha^2+(1-object$other$alpha)^2)),
                                                         object$other$alpha),
                                   "dlnorm"=rlnorm(h*nsim, -object$scale^2/2, object$scale)-1,
                                   "dinvgauss"=rinvgauss(h*nsim, 1, dispersion=sigmaValue^2)-1,
                                   "dllaplace"=exp(rlaplace(h*nsim, 0, sigmaValue/2))-1,
                                   "dls"=exp(rs(h*nsim, 0, (sigmaValue^2/120)^0.25))-1,
                                   "dlgnorm"=exp(rgnorm(h*nsim, 0,
                                                        sigmaValue*sqrt(gamma(1/object$other$beta)/gamma(3/object$other$beta))))-1
                                   ),
                            h,nsim);
        # Normalise errors in order not to get ridiculous things on small nsim
        if(nsim<=500){
            if(Etype=="A"){
                matErrors[] <- matErrors - array(apply(matErrors,1,mean),c(h,nsim));
            }
            else{
                matErrors[] <- (1+matErrors) / array(apply(1+matErrors,1,mean),c(h,nsim))-1;
            }
        }
        # This stuff is needed in order to produce adequate values for weird models
        EtypeModified <- Etype;
        if(Etype=="A" && any(object$distribution==c("dlnorm","dinvgauss","dls","dllaplace"))){
            EtypeModified[] <- "M";
        }

        # States, Errors, Ot, Transition, Measurement, Persistence
        ySimulated <- adamSimulatorWrap(arrVt, matErrors,
                                        # matrix(rep(1,h*nsim), h, nsim),
                                        matrix(rbinom(h*nsim, 1, pForecast), h, nsim),
                                        array(matF,c(dim(matF),nsim)), matWt,
                                        matrix(vecG, componentsNumberETS+componentsNumberARIMA+xregNumber, nsim),
                                        EtypeModified, Ttype, Stype,
                                        lagsModelAll, profilesObservedTable, profilesRecentTable,
                                        componentsNumberETSSeasonal, componentsNumberETS,
                                        componentsNumberARIMA, xregNumber)$matrixYt;

        #### Note that the cumulative doesn't work with oes at the moment!
        if(cumulative){
            yForecast[] <- mean(colSums(ySimulated,na.rm=T));
            yLower[] <- quantile(colSums(ySimulated,na.rm=T),levelLow,type=7);
            yUpper[] <- quantile(colSums(ySimulated,na.rm=T),levelUp,type=7);
        }
        else{
            for(i in 1:h){
                if(Ttype=="M" || Stype=="M"){
                    yForecast[i] <- mean(ySimulated[i,],na.rm=T);
                }
                yLower[i,] <- quantile(ySimulated[i,],levelLow[i,],na.rm=T,type=7);
                yUpper[i,] <- quantile(ySimulated[i,],levelUp[i,],na.rm=T,type=7);
            }
        }
        # This step is needed in order to make intervals similar between the different methods
        if(Etype=="A"){
            yLower[] <- yLower - yForecast;
            yUpper[] <- yUpper - yForecast;
        }
        else{
            yLower[] <- yLower / yForecast;
            yUpper[] <- yUpper / yForecast;
        }
    }
    else{
        #### Approximate and confidence interval ####
        # Produce covatiance matrix and use it
        if(any(interval==c("approximate","confidence"))){
            s2 <- sigma(object)^2;
            # IG and Lnorm can use approximations from the multiplications
            if(any(object$distribution==c("dinvgauss","dlnorm","dllaplace","dls","dlgnorm")) && Etype=="M"){
                vcovMulti <- adamVarAnal(lagsModelAll, h, matWt[1,,drop=FALSE], matF, vecG, s2);
                if(any(object$distribution==c("dlnorm","dls","dllaplace","dlgnorm"))){
                    vcovMulti[] <- log(1+vcovMulti);
                }

                # The confidence interval relies on the assumption that initial level is known
                if(interval=="confidence"){
                    vcovMulti[] <- vcovMulti - s2;
                }

                # We don't do correct cumulatives in this case...
                if(cumulative){
                    vcovMulti <- sum(vcovMulti);
                }
            }
            else{
                vcovMulti <- covarAnal(lagsModelAll, h, matWt[1,,drop=FALSE], matF, vecG, s2);

                # The confidence interval relies on the assumption that initial level is known
                if(interval=="confidence"){
                    vcovMulti[] <- vcovMulti - s2;
                }

                # Do either the variance of sum, or a diagonal
                if(cumulative){
                    vcovMulti <- sum(vcovMulti);
                }
                else{
                    vcovMulti <- diag(vcovMulti);
                }
            }
        }
        #### Semiparametric and nonparametric interval ####
        # Extract multistep errors and calculate the covariance matrix
        else if(any(interval==c("semiparametric","nonparametric"))){
            if(h>1){
                adamErrors <- as.matrix(rmultistep(object, h=h));

                if(any(object$distribution==c("dinvgauss","dlnorm","dls","dllaplace","dlgnorm")) && (Etype=="A")){
                    yFittedMatrix <- adamErrors;
                    for(i in 1:h){
                        yFittedMatrix[,i] <- fitted(object)[1:(obsInSample-h)+i];
                    }
                    adamErrors[] <- adamErrors/yFittedMatrix;
                }

                if(interval=="semiparametric"){
                    # Do either the variance of sum, or a diagonal
                    if(cumulative){
                        vcovMulti <- sum(t(adamErrors) %*% adamErrors / (obsInSample-h));
                    }
                    else{
                        vcovMulti <- diag(t(adamErrors) %*% adamErrors / (obsInSample-h));
                    }
                }
                # For nonparametric and cumulative...
                else{
                    if(cumulative){
                        adamErrors <- matrix(apply(adamErrors, 2, sum),obsInSample-h,1);
                    }
                }
            }
            else{
                vcovMulti <- sigma(object)^2;
                adamErrors <- as.vector(residuals(object));
            }
        }

        # Calculate interval for approximate and semiparametric
        if(any(interval==c("approximate","confidence","semiparametric"))){
            if(object$distribution=="dnorm"){
                if(Etype=="A"){
                    yLower[] <- qnorm(levelLow, 0, sqrt(vcovMulti));
                    yUpper[] <- qnorm(levelUp, 0, sqrt(vcovMulti));
                }
                else{
                    yLower[] <- qnorm(levelLow, 1, sqrt(vcovMulti));
                    yUpper[] <- qnorm(levelUp, 1, sqrt(vcovMulti));
                }
            }
            else if(object$distribution=="dlaplace"){
                if(Etype=="A"){
                    yLower[] <- qlaplace(levelLow, 0, sqrt(vcovMulti/2));
                    yUpper[] <- qlaplace(levelUp, 0, sqrt(vcovMulti/2));
                }
                else{
                    yLower[] <- qlaplace(levelLow, 1, sqrt(vcovMulti/2));
                    yUpper[] <- qlaplace(levelUp, 1, sqrt(vcovMulti/2));
                }
            }
            else if(object$distribution=="ds"){
                if(Etype=="A"){
                    yLower[] <- qs(levelLow, 0, (vcovMulti/120)^0.25);
                    yUpper[] <- qs(levelUp, 0, (vcovMulti/120)^0.25);
                }
                else{
                    yLower[] <- qs(levelLow, 1, (vcovMulti/120)^0.25);
                    yUpper[] <- qs(levelUp, 1, (vcovMulti/120)^0.25);
                }
            }
            else if(object$distribution=="dgnorm"){
                alpha <- sqrt(vcovMulti*(gamma(1/object$other$beta)/gamma(3/object$other$beta)));
                if(Etype=="A"){
                    yLower[] <- suppressWarnings(qgnorm(levelLow, 0, alpha, object$other$beta));
                    yUpper[] <- suppressWarnings(qgnorm(levelUp, 0, alpha, object$other$beta));
                }
                else{
                    yLower[] <- suppressWarnings(qgnorm(levelLow, 1, alpha, object$other$beta));
                    yUpper[] <- suppressWarnings(qgnorm(levelUp, 1, alpha, object$other$beta));
                }
            }
            else if(object$distribution=="dlogis"){
                if(Etype=="A"){
                    yLower[] <- qlogis(levelLow, 0, sqrt(vcovMulti*3)/pi);
                    yUpper[] <- qlogis(levelUp, 0, sqrt(vcovMulti*3)/pi);
                }
                else{
                    yLower[] <- qlogis(levelLow, 1, sqrt(vcovMulti*3)/pi);
                    yUpper[] <- qlogis(levelUp, 1, sqrt(vcovMulti*3)/pi);
                }
            }
            else if(object$distribution=="dt"){
                df <- nobs(object) - nparam(object);
                if(Etype=="A"){
                    yLower[] <- sqrt(vcovMulti)*qt(levelLow, df);
                    yUpper[] <- sqrt(vcovMulti)*qt(levelUp, df);
                }
                else{
                    yLower[] <- (1 + sqrt(vcovMulti)*qt(levelLow, df));
                    yUpper[] <- (1 + sqrt(vcovMulti)*qt(levelUp, df));
                }
            }
            else if(object$distribution=="dalaplace"){
                alpha <- object$other$alpha;
                if(Etype=="A"){
                    yLower[] <- qalaplace(levelLow, 0,
                                          sqrt(vcovMulti*alpha^2*(1-alpha)^2/(alpha^2+(1-alpha)^2)), alpha);
                    yUpper[] <- qalaplace(levelUp, 0,
                                          sqrt(vcovMulti*alpha^2*(1-alpha)^2/(alpha^2+(1-alpha)^2)), alpha);
                }
                else{
                    yLower[] <- qalaplace(levelLow, 1,
                                          sqrt(vcovMulti*alpha^2*(1-alpha)^2/(alpha^2+(1-alpha)^2)), alpha);
                    yUpper[] <- qalaplace(levelUp, 1,
                                          sqrt(vcovMulti*alpha^2*(1-alpha)^2/(alpha^2+(1-alpha)^2)), alpha);
                }
            }
            else if(object$distribution=="dlnorm"){
                yLower[] <- qlnorm(levelLow, Re(1-sqrt(abs(1-vcovMulti))),
                                   sqrt(2*Re(1-sqrt(abs(1-vcovMulti)))));
                yUpper[] <- qlnorm(levelUp, Re(1-sqrt(abs(1-vcovMulti))),
                                   sqrt(2*Re(1-sqrt(abs(1-vcovMulti)))));
                if(Etype=="A"){
                    yLower[] <- (yLower-1)*yForecast;
                    yUpper[] <-(yUpper-1)*yForecast;
                }
            }
            else if(object$distribution=="dllaplace"){
                yLower[] <- exp(qlaplace(levelLow, 0, sqrt(vcovMulti/2)));
                yUpper[] <- exp(qlaplace(levelUp, 0, sqrt(vcovMulti/2)));
                if(Etype=="A"){
                    yLower[] <- (yLower-1)*yForecast;
                    yUpper[] <-(yUpper-1)*yForecast;
                }
            }
            else if(object$distribution=="dls"){
                yLower[] <- exp(qs(levelLow, 0, (vcovMulti/120)^0.25));
                yUpper[] <- exp(qs(levelUp, 0, (vcovMulti/120)^0.25));
                if(Etype=="A"){
                    yLower[] <- (yLower-1)*yForecast;
                    yUpper[] <-(yUpper-1)*yForecast;
                }
            }
            else if(object$distribution=="dlgnorm"){
                alpha <- sqrt(vcovMulti*(gamma(1/object$other$beta)/gamma(3/object$other$beta)));
                yLower[] <- suppressWarnings(exp(qgnorm(levelLow, 0, alpha, object$other$beta)));
                yUpper[] <- suppressWarnings(exp(qgnorm(levelUp, 0, alpha, object$other$beta)));
            }
            else if(object$distribution=="dinvgauss"){
                yLower[] <- qinvgauss(levelLow, 1, dispersion=vcovMulti);
                yUpper[] <- qinvgauss(levelUp, 1, dispersion=vcovMulti);
                if(Etype=="A"){
                    yLower[] <- (yLower-1)*yForecast;
                    yUpper[] <-(yUpper-1)*yForecast;
                }
            }
        }
        # Use Taylor & Bunn approach for the nonparametric ones
        #### Nonparametric intervals, regression ####
        else if(interval=="nonparametric"){
            if(h>1){
                # This is needed in order to see if quant regression can be used
                if(all(levelLow==levelLow[1,])){
                    levelLow <- levelLow[1,,drop=FALSE];
                }
                if(all(levelUp==levelUp[1,])){
                    levelUp <- levelUp[1,,drop=FALSE];
                }

                # Do quantile regression for h>1 and scalars for the level (no change across h)
                # transpose is needed in order to compare correctly
                if(all(t(levelNew)==levelNew[1,])){
                    # Quantile regression function
                    intervalQuantile <- function(A, alpha){
                        ee[] <- adamErrors - (A[1]*xe^A[2]);
                        return((1-alpha)*sum(abs(ee[ee<0]))+alpha*sum(abs(ee[ee>=0])));
                    }

                    ee <- adamErrors;
                    xe <- matrix(c(1:h),nrow=nrow(ee),ncol=ncol(ee),byrow=TRUE);

                    for(i in 1:nLevels){
                        # lower quantiles
                        A <- nlminb(rep(1,2),intervalQuantile,alpha=levelLow[1,i])$par;
                        yLower[,i] <- A[1]*c(1:h)^A[2];

                        # upper quantiles
                        A[] <- nlminb(rep(1,2),intervalQuantile,alpha=levelUp[1,i])$par;
                        yUpper[,i] <- A[1]*c(1:h)^A[2];
                    }
                }
                # Otherwise just return quantiles of errors
                else{
                    if(cumulative){
                        yLower[] <- quantile(adamErrors,levelLow,type=7);
                        yUpper[] <- quantile(adamErrors,levelUp,type=7);
                    }
                    else{
                        for(i in 1:h){
                            yLower[i] <- quantile(adamErrors[,i],levelLow[i],na.rm=T,type=7);
                            yUpper[i] <- quantile(adamErrors[,i],levelUp[i],na.rm=T,type=7);
                        }
                    }
                }
            }
            else{
                yLower[] <- quantile(adamErrors,levelLow,type=7);
                yUpper[] <- quantile(adamErrors,levelUp,type=7);
            }

            if(Etype=="M"){
                yLower[] <- 1+yLower;
                yUpper[] <- 1+yUpper;
            }
            else if(Etype=="A" & any(object$distribution==c("dinvgauss","dlnorm","dllaplace","dls","dlgnorm"))){
                yLower[] <- yLower*yForecast;
                yUpper[] <- yUpper*yForecast;
            }
        }
        else{
            yUpper[] <- yLower[] <- NA;
        }
    }

    # Fix of prediction intervals depending on what has happened
    if(interval!="none"){
        # Make sensible values out of those weird quantiles
        if(!cumulative){
            if(any(levelLow==0)){
                # zoo does not like, when you work with matrices of indices... silly thing
                yBoundBuffer <- levelLow;
                yBoundBuffer[] <- yLower
                if(Etype=="A"){
                    yBoundBuffer[levelLow==0] <- -Inf;
                    yLower[] <- yBoundBuffer;
                }
                else{
                    yBoundBuffer[levelLow==0] <- 0;
                    yLower[] <- yBoundBuffer;
                }
            }
            if(any(levelUp==1)){
                # zoo does not like, when you work with matrices of indices... silly thing
                yBoundBuffer <- levelUp;
                yBoundBuffer[] <- yUpper
                yBoundBuffer[levelUp==1] <- Inf;
                yUpper[] <- yBoundBuffer;
            }
        }
        else{
            if(Etype=="A" && any(levelLow==0)){
                yLower[] <- -Inf;
            }
            else if(Etype=="M" && any(levelLow==0)){
                yLower[] <- 0;
            }
            if(any(levelUp==1)){
                yUpper[] <- Inf;
            }
        }

        # Substitute NAs and NaNs with zeroes
        if(any(is.nan(yLower)) || any(is.na(yLower))){
            yLower[is.nan(yLower)] <- switch(Etype,"A"=0,"M"=1);
            yLower[is.na(yLower)] <- switch(Etype,"A"=0,"M"=1);
        }
        if(any(is.nan(yUpper)) || any(is.na(yUpper))){
            yUpper[is.nan(yUpper)] <- switch(Etype,"A"=0,"M"=1);
            yUpper[is.na(yUpper)] <- switch(Etype,"A"=0,"M"=1);
        }

        # Do intervals around the forecasts...
        if(Etype=="A"){
            yLower[] <- yForecast + yLower;
            yUpper[] <- yForecast + yUpper;
        }
        else{
            yLower[] <- yForecast*yLower;
            yUpper[] <- yForecast*yUpper;
        }

        # Check what we have from the occurrence model
        if(occurrenceModel){
            # If there are NAs, then there's no variability and no intervals.
            if(any(is.na(yUpper))){
                yUpper[is.na(yUpper)] <- (yForecast/pForecast)[is.na(yUpper)];
            }
            if(any(is.na(yLower))){
                yLower[is.na(yLower)] <- 0;
            }
        }

        colnames(yLower) <- switch(side,
                                   "both"=paste0("Lower bound (",(1-level)/2*100,"%)"),
                                   "lower"=paste0("Lower bound (",(1-level)*100,"%)"),
                                   "upper"=rep("Lower 0%",nLevels));

        colnames(yUpper) <- switch(side,
                                   "both"=paste0("Upper bound (",(1+level)/2*100,"%)"),
                                   "lower"=rep("Upper 100%",nLevels),
                                   "upper"=paste0("Upper bound (",level*100,"%)"));
    }

    return(structure(list(mean=yForecast, lower=yLower, upper=yUpper, model=object,
                          level=level, interval=interval, side=side, cumulative=cumulative),
                     class=c("adam.forecast","smooth.forecast","forecast")));
}

#' @export
forecast.adamCombined <- function(object, h=10, newdata=NULL,
                                  interval=c("none", "simulated", "approximate", "semiparametric", "nonparametric"),
                                  level=0.95, side=c("both","upper","lower"), cumulative=FALSE, nsim=5000, ...){

    interval <- match.arg(interval[1],c("none", "simulated", "approximate", "semiparametric",
                                        "nonparametric", "confidence", "parametric","prediction"));
    side <- match.arg(side);

    yClasses <- class(actuals(object));
    obsInSample <- nobs(object);

    if(any(yClasses=="ts")){
        # ts structure
        yForecastStart <- time(actuals(object))[obsInSample]+deltat(actuals(object));
        yFrequency <- frequency(actuals(object));
    }
    else{
        # zoo thingy
        yIndex <- time(actuals(object));
        yForecastIndex <- yIndex[obsInSample]+diff(tail(yIndex,2))*c(1:h);
    }

    # How many levels did user asked to produce
    nLevels <- length(level);
    # Cumulative forecasts have only one observation
    if(cumulative){
        # hFinal is the number of elements we will have in the final forecast
        hFinal <- 1;
    }
    else{
        hFinal <- h;
    }

    # Create necessary matrices for the forecasts
    if(any(yClasses=="ts")){
        yForecast <- ts(vector("numeric", hFinal), start=yForecastStart, frequency=yFrequency);
        yUpper <- yLower <- ts(matrix(0,hFinal,nLevels), start=yForecastStart, frequency=yFrequency);
    }
    else{
        yForecast <- zoo(vector("numeric", hFinal), order.by=yForecastIndex);
        yUpper <- yLower <- zoo(matrix(0,hFinal,nLevels), order.by=yForecastIndex);
    }

    # The list contains 8 elements
    adamForecasts <- vector("list",8);
    names(adamForecasts)[c(1:3)] <- c("mean","lower","upper");
    for(i in 1:length(object$models)){
        adamForecasts[] <- forecast.adam(object$models[[i]], h=h, newdata=newdata,
                                         interval=interval,
                                         level=level, side=side, cumulative=cumulative, nsim=nsim, ...);
        yForecast[] <- yForecast + adamForecasts$mean * object$ICw[i];
        yUpper[] <- yUpper + adamForecasts$upper * object$ICw[i];
        yLower[] <- yLower + adamForecasts$lower * object$ICw[i];
    }

    # Fix the names of the columns
    if(interval!="none"){
        colnames(yLower) <- colnames(adamForecasts$lower);
        colnames(yUpper) <- colnames(adamForecasts$upper);
    }

    # Fix the content of upper / lower bounds
    if(side=="upper"){
        yLower[] <- -Inf;
    }
    else if(side=="lower"){
        yUpper[] <- Inf;
    }

    # Get rid of specific models
    object$models <- NULL;

    return(structure(list(mean=yForecast, lower=yLower, upper=yUpper, model=object,
                          level=level, interval=interval, side=side, cumulative=cumulative),
                     class=c("adam.forecast","smooth.forecast","forecast")));
}

#' @export
print.adam.forecast <- function(x, ...){
    if(x$interval!="none"){
        returnedValue <- switch(x$side,
                                "both"=cbind(x$mean,x$lower,x$upper),
                                "lower"=cbind(x$mean,x$lower),
                                "upper"=cbind(x$mean,x$upper));
        colnames(returnedValue) <- switch(x$side,
                                          "both"=c("Point forecast",colnames(x$lower),colnames(x$upper)),
                                          "lower"=c("Point forecast",colnames(x$lower)),
                                          "upper"=c("Point forecast",colnames(x$upper)))
    }
    else{
        returnedValue <- x$mean;
    }
    print(returnedValue);
}

#' @export
plot.adam.forecast <- function(x, ...){
    yClasses <- class(actuals(x));
    digits <- 2;

    ellipsis <- list(...);
    if(is.null(ellipsis$ylim)){
        vectorOfValues <- switch(x$side,
                                 "both"=c(as.vector(actuals(x$model)),as.vector(x$mean),
                                          as.vector(x$lower),as.vector(x$upper)),
                                 "lower"=c(as.vector(actuals(x$model)),as.vector(x$mean),
                                           as.vector(x$lower)),
                                 "upper"=c(as.vector(actuals(x$model)),as.vector(x$mean),
                                           as.vector(x$upper)));
        ellipsis$ylim <- range(vectorOfValues[is.finite(vectorOfValues)],na.rm=TRUE);
    }

    if(is.null(ellipsis$legend)){
        ellipsis$legend <- FALSE;
        ellipsis$parReset <- FALSE;
    }

    if(is.null(ellipsis$main)){
        distrib <- switch(x$model$distribution,
                          "dnorm" = "Normal",
                          "dlogis" = "Logistic",
                          "dlaplace" = "Laplace",
                          "ds" = "S",
                          "dgnorm" = paste0("Generalised Normal with beta=",round(x$model$other$beta,digits)),
                          "dalaplace" = paste0("Asymmetric Laplace with alpha=",round(x$model$other$alpha,digits)),
                          "dt" = paste0("Student t with nu=",round(x$model$other$nu, digits)),
                          "dlnorm" = "Log Normal",
                          "dllaplace" = "Log Laplace",
                          "dls" = "Log S",
                          "dgnorm" = paste0("Log Generalised Normal with beta=",round(x$model$other$beta,digits)),
                          # "dbcnorm" = paste0("Box-Cox Normal with lambda=",round(x$other$lambda,2)),
                          "dinvgauss" = "Inverse Gaussian",
                          "default"
        );
        ellipsis$main <- paste0("Forecast from ",x$model$model," with ",distrib," distribution");
    }

    if(!is.null(x$model$holdout)){
        if(any(yClasses=="ts")){
            ellipsis$actuals <- ts(c(actuals(x$model),x$model$holdout[,1]),
                                   start=start(actuals(x$model)),
                                   frequency=frequency(actuals(x$model)));
        }
        else{
            ellipsis$actuals <- zoo(c(as.vector(actuals(x$model)),as.vector(x$model$holdout[,1])),
                                    order.by=c(time(actuals(x$model)),time(x$model$holdout)));
        }
    }
    else{
        ellipsis$actuals <- actuals(x$model);
    }

    ellipsis$forecast <- x$mean;
    ellipsis$fitted <- fitted(x);
    ellipsis$lower <- x$lower;
    ellipsis$upper <- x$upper;
    ellipsis$level <- x$level;

    do.call(graphmaker, ellipsis);
}


#### Refitter and reforecaster ####
#' Refit the model with randomly generated initial parameters and produce forecasts
#'
#' \code{refit} function generates the parameters based on the values in the provided
#' object and then reapplies the same model with those parameters to the data, getting
#' the fitted paths and updated states. \code{reforecast} function uses those values
#' in order to produce forecasts for the \code{h} steps ahead.
#'
#' The main motivation of the function is to take the randomness due to the in-sample
#' estimation of parameters into account when fitting the model and to propagate
#' this randomness to the forecasts. The methods can be considered as a special case
#' of recursive bootstrap.
#'
#' @template ssAuthor
#' @template ssKeywords
#'
#' @param object Model estimated using one of the functions of smooth package.
#' @param nsim Number of paths to generate (number of simulations to do).
#' @param h Forecast horizon.
#' @param newdata The new data needed in order to produce forecasts.
#' @param occurrence The vector containing the future occurrence variable
#' (values in [0,1]), if it is known.
#' @param interval What type of mechanism to use for interval construction. The options
#' include \code{interval="none"}, \code{interval="prediction"} (prediction intervals)
#' and \code{interval="confidence"} (intervals for the point forecast). The other options
#' are not supported and do not make much sense for the refitted model.
#' @param level Confidence level. Defines width of prediction interval.
#' @param side Defines, whether to provide \code{"both"} sides of prediction
#' interval or only \code{"upper"}, or \code{"lower"}.
#' @param cumulative If \code{TRUE}, then the cumulative forecast and prediction
#' interval are produced instead of the normal ones. This is useful for
#' inventory control systems.
#' @param ... Other parameters passed to \code{mean()} function in case of
#' \code{reforecast} (this mainly refers to \code{trim} variable, which is set to
#' 0.01 by default) and to \code{vcov} in case of \code{refit}.
#' @return \code{refit()} returns object of the class "refit", which contains:
#' \itemize{
#' \item \code{states} - The array of states of the model;
#' \item \code{fitted} - The matrix with fitted values, where columns correspond
#' to different paths;
#' \item \code{transition} - The array of transition matrices;
#' \item \code{measurement} - The array of measurement matrices;
#' \item \code{persistence} - The matrix of persistence vectors (paths in columns);
#' \item \code{profile} - The array of profiles obtained by the end of each fit.
#' }
#'
#' \code{reforecast()} returns the object of the class \link[smooth]{forecast.smooth},
#' which contains in addition to the standard list the variable \code{paths} - all
#' simulated trajectories with h in rows, simulated future paths for each state in
#' columns and different states (obtained from \code{refit()} function) in the
#' third dimension.
#'
#' @seealso \link[smooth]{forecast.smooth}
#' @examples
#'
#' x <- rnorm(100,0,1)
#'
#' # Just as example. orders and lags do not return anything for ces() and es(). But modelType() does.
#' ourModel <- adam(x, "ANN")
#' refittedModel <- refit(ourModel, nsim=50)
#' plot(refittedModel)
#'
#' ourForecast <- reforecast(ourModel, nsim=50)
#'
#' @rdname refit
#' @export refit
refit <- function(object, nsim=1000, ...) UseMethod("refit")

#' @export
refit.default <- function(object, nsim=1000, ...){
    warning(paste0("The method is not implemented for the object of the class ",class(object)[1]),
            call.=FALSE);
    return(structure(list(states=object$states, fitted=fitted(object)),
                     class="refit"));
}

#' @importFrom MASS mvrnorm
#' @export
refit.adam <- function(object, nsim=1000, ...){
    # Start measuring the time of calculations
    startTime <- Sys.time();

    vcovAdam <- suppressWarnings(vcov(object, ...));
    parametersNames <- colnames(vcovAdam);
    # Check if the matrix is positive definite
    vcovEigen <- min(eigen(vcovAdam, only.values=TRUE)$values);
    if(vcovEigen<=0){
        if(vcovEigen>-1){
            warning(paste0("The covariance matrix of parameters is not positive semi-definite. ",
                           "We will try fixing this, but it might make sense re-estimating adam(), tuning the optimiser."),
                    call.=FALSE, immediate.=TRUE);
            # Tune the thing a bit - one of simple ways to fix the issue
            epsilon <- -vcovEigen+1e-10;
            vcovAdam[] <- vcovAdam + epsilon*diag(nrow(vcovAdam));
        }
        else{
            warning(paste0("The covariance matrix of parameters is not positive semi-definite. ",
                           "We cannot fix it, so we will use the diagonal only. ",
                           "It makes sense to re-estimate adam(), tuning the optimiser. ",
                           "For example, try reoptimising via 'object <- adam(y, ..., B=object$B)'."),
                    call.=FALSE, immediate.=TRUE);
            vcovAdam[] <- diag(diag(vcovAdam));
        }
    }

    # All the variables needed in the refitter
    yInSample <- actuals(object);
    yClasses <- class(yInSample);
    parametersNumber <- length(parametersNames);
    obsInSample <- nobs(object);
    Etype <- errorType(object);
    Ttype <- substr(modelType(object),2,2);
    Stype <- substr(modelType(object),nchar(modelType(object)),nchar(modelType(object)));
    etsModel <- any(unlist(gregexpr("ETS",object$model))!=-1);
    arimaModel <- any(unlist(gregexpr("ARIMA",object$model))!=-1);
    lags <- object$lags;
    lagsSeasonal <- lags[lags!=1];
    lagsModelAll <- object$lagsAll;
    lagsModelMax <- max(lagsModelAll);
    persistence <- as.matrix(object$persistence);
    # If there is xreg, but no deltas, increase persistence by including zeroes
    if(ncol(object$data)>1 && !any(substr(names(object$persistence),1,5)=="delta")){
        persistence <- rbind(persistence,matrix(rep(0,sum(object$nParam[,2])),ncol=1));
    }

    if(!is.null(object$initial$seasonal)){
        if(is.list(object$initial$seasonal)){
            componentsNumberETSSeasonal <- length(object$initial$seasonal);
        }
        else{
            componentsNumberETSSeasonal <- 1;
        }
    }
    else{
        componentsNumberETSSeasonal <- 0;
    }
    componentsNumberETS <- length(object$initial$level) + length(object$initial$trend) + componentsNumberETSSeasonal;
    componentsNumberARIMA <- sum(substr(colnames(object$states),1,10)=="ARIMAState");
    if(!is.null(object$initial$xreg)){
        xregModel <- TRUE;

        #### Create xreg vectors ####
        xreg <- object$data;
        formula <- formula(object)
        responseName <- all.vars(formula)[1];
        # Robustify the names of variables
        colnames(xreg) <- make.names(colnames(xreg),unique=TRUE);
        # The names of the original variables
        xregNamesOriginal <- all.vars(formula)[-1];
        # Levels for the factors
        xregFactorsLevels <- lapply(xreg,levels);
        xregFactorsLevels[[responseName]] <- NULL;
        # Expand the variables. We cannot use alm, because it is based on obsInSample
        xregData <- model.frame(formula,data=as.data.frame(xreg));
        # Binary, flagging factors in the data
        xregFactors <- (attr(terms(xregData),"dataClasses")=="factor")[-1];
        # Get the names from the standard model.matrix
        xregNames <- colnames(model.matrix(xregData,data=xregData));
        # Expanded stuff with all levels for factors
        if(any(xregFactors)){
            xregModelMatrix <- model.matrix(xregData,xregData,
                                            contrasts.arg=lapply(xregData[attr(terms(xregData),"dataClasses")=="factor"],
                                                                 contrasts, contrasts=FALSE));
            xregNamesModified <- colnames(xregModelMatrix)[-1];
        }
        else{
            xregModelMatrix <- model.matrix(xregData,data=xregData);
            xregNamesModified <- xregNames;
        }
        xregData <- as.matrix(xregModelMatrix);
        # Remove intercept
        interceptIsPresent <- FALSE;
        if(any(colnames(xregData)=="(Intercept)")){
            interceptIsPresent[] <- TRUE;
            xregData <- xregData[,-1,drop=FALSE];
        }
        xregNumber <- ncol(xregData);

        # The indices of the original parameters
        xregParametersMissing <- setNames(vector("numeric",xregNumber),xregNamesModified);
        # # The indices of the original parameters
        xregParametersIncluded <- setNames(vector("numeric",xregNumber),xregNamesModified);
        # The vector, marking the same values of smoothing parameters
        if(interceptIsPresent){
            xregParametersPersistence <- setNames(attr(xregModelMatrix,"assign")[-1],xregNamesModified);
        }
        else{
            xregParametersPersistence <- setNames(attr(xregModelMatrix,"assign"),xregNamesModified);
        }

        # If there are factors not in the alm data, create additional initials
        if(any(!(xregNamesModified %in% xregNames))){
            xregAbsent <- !(xregNamesModified %in% xregNames);
            # Go through new names and find, where they came from. Then get the missing parameters
            for(i in which(xregAbsent)){
                # Find the name of the original variable
                # Use only the last value... hoping that the names like x and x1 are not used.
                xregNameFound <- tail(names(sapply(xregNamesOriginal,grepl,xregNamesModified[i])),1);
                # Get the indices of all k-1 levels
                xregParametersIncluded[xregNames[xregNames %in% paste0(xregNameFound,
                                                                       xregFactorsLevels[[xregNameFound]])]] <- i;
                # Get the index of the absent one
                xregParametersMissing[i] <- i;
            }
            # Write down the new parameters
            xregNames <- xregNamesModified;
        }
        # The vector of parameters that should be estimated (numeric + original levels of factors)
        xregParametersEstimated <- xregParametersIncluded
        xregParametersEstimated[xregParametersEstimated!=0] <- 1;
        xregParametersEstimated[xregParametersMissing==0 & xregParametersIncluded==0] <- 1;
    }
    else{
        xregModel <- FALSE;
        xregNumber <- 0;
        xregParametersMissing <- 0;
        xregParametersIncluded <- 0;
        xregParametersEstimated <- 0;
        xregParametersPersistence <- 0;
    }
    profilesObservedTable <- adamProfileCreator(lagsModelAll, lagsModelMax, obsInSample)$observed;

    # Generate the data from the multivariate normal
    randomParameters <- mvrnorm(nsim, coef(object), vcovAdam);

    #### Rectify the random values for smoothing parameters ####
    if(etsModel){
        # Usual bounds
        if(object$bounds=="usual"){
            # Set the bounds for alpha
            if(any(parametersNames=="alpha")){
                randomParameters[randomParameters[,"alpha"]<0,"alpha"] <- 0;
                randomParameters[randomParameters[,"alpha"]>1,"alpha"] <- 1;
            }
            # Set the bounds for beta
            if(any(parametersNames=="beta")){
                randomParameters[randomParameters[,"beta"]<0,"beta"] <- 0;
                randomParameters[randomParameters[,"beta"]>randomParameters[,"alpha"],"beta"] <-
                    randomParameters[randomParameters[,"beta"]>randomParameters[,"alpha"],"alpha"];
            }
            # Set the bounds for gamma
            if(any(substr(parametersNames,1,5)=="gamma")){
                gammas <- which(substr(colnames(randomParameters),1,5)=="gamma");
                for(i in 1:length(gammas)){
                    randomParameters[randomParameters[,gammas[i]]<0,gammas[i]] <- 0;
                    randomParameters[randomParameters[,gammas[i]]>randomParameters[,"alpha"],
                                     gammas[i]] <- 1-
                        randomParameters[randomParameters[,gammas[i]]>randomParameters[,"alpha"],"alpha"];
                }
            }
            # Set the bounds for phi
            if(any(parametersNames=="phi")){
                randomParameters[randomParameters[,"phi"]<0,"phi"] <- 0;
                randomParameters[randomParameters[,"phi"]>1,"phi"] <- 1;
            }
        }
        # Admissible bounds
        else if(object$bounds=="admissible"){
            # Check, if there is alpha
            if(any(parametersNames=="alpha")){
                alphaBounds <- eigenBounds(object, persistence,
                                           variableNumber=which(names(object$persistence)=="alpha"));
                randomParameters[randomParameters[,"alpha"]<alphaBounds[1],"alpha"] <- alphaBounds[1];
                randomParameters[randomParameters[,"alpha"]>alphaBounds[2],"alpha"] <- alphaBounds[2];
            }
            # Check, if there is beta
            if(any(parametersNames=="beta")){
                betaBounds <- eigenBounds(object, persistence,
                                          variableNumber=which(names(object$persistence)=="beta"));
                randomParameters[randomParameters[,"beta"]<betaBounds[1],"beta"] <- betaBounds[1];
                randomParameters[randomParameters[,"beta"]>betaBounds[2],"beta"] <- betaBounds[2];
            }
            # Check, if there are gammas
            if(any(substr(parametersNames,1,5)=="gamma")){
                gammas <- which(substr(parametersNames,1,5)=="gamma");
                for(i in 1:length(gammas)){
                    gammaBounds <- eigenBounds(object, persistence,
                                               variableNumber=which(substr(names(object$persistence),1,5)=="gamma")[i]);
                    randomParameters[randomParameters[,gammas[i]]<gammaBounds[1],gammas[i]] <- gammaBounds[1];
                    randomParameters[randomParameters[,gammas[i]]>gammaBounds[2],gammas[i]] <- gammaBounds[2];
                }
            }
            # Check, if there are deltas (for xreg)
            if(any(substr(parametersNames,1,5)=="delta")){
                deltas <- which(substr(parametersNames,1,5)=="delta");
                for(i in 1:length(deltas)){
                    deltaBounds <- eigenBounds(object, persistence,
                                               variableNumber=which(substr(names(object$persistence),1,5)=="delta")[i]);
                    randomParameters[randomParameters[,deltas[i]]<deltaBounds[1],deltas[i]] <- deltaBounds[1];
                    randomParameters[randomParameters[,deltas[i]]>deltaBounds[2],deltas[i]] <- deltaBounds[2];
                }
            }
        }

        # States
        # Set the bounds for trend
        if(Ttype=="M" && any(parametersNames=="trend")){
            randomParameters[randomParameters[,"trend"]<0,"trend"] <- 1e-6;
        }
        # Seasonality
        if(Stype=="M" && any(substr(parametersNames,1,8)=="seasonal")){
            seasonals <- which(substr(parametersNames,1,8)=="seasonal");
            for(i in seasonals){
                randomParameters[randomParameters[,i]<0,i] <- 1e-6;
            }
        }
    }

    # Correct the bounds for the ARIMA model
    if(arimaModel){
        #### Deal with ARIMA parameters ####
        ariPolynomial <- object$other$polynomial$ariPolynomial;
        arPolynomial <- object$other$polynomial$arPolynomial;
        maPolynomial <- object$other$polynomial$maPolynomial;
        nonZeroARI <- object$other$ARIMAIndices$nonZeroARI;
        nonZeroMA <- object$other$ARIMAIndices$nonZeroMA;
        arPolynomialMatrix <- object$other$arPolynomialMatrix;
        # Locate all thetas for ARIMA
        thetas <- which(substr(parametersNames,1,5)=="theta");
        # Locate phi for ARIMA (they are always phi1, phi2 etc)
        phis <- which((substr(parametersNames,1,3)=="phi") & (nchar(parametersNames)>3));
        # Do loop for thetas
        if(length(thetas)>0){
            # MA parameters
            for(i in 1:length(thetas)){
                psiBounds <- eigenBounds(object, persistence,
                                         variableNumber=which(substr(names(object$persistence),1,3)=="psi")[nonZeroMA[i,2]]);
                # If there are ARI elements in persistence, subtract (-(-x)) them to get proper bounds
                if(any(nonZeroARI[,2]==i)){
                    ariIndex <- which(nonZeroARI[,2]==i);
                    randomParameters[randomParameters[,thetas[i]]-ariPolynomial[nonZeroARI[ariIndex,1]]<psiBounds[1],thetas[i]] <-
                        psiBounds[1]+ariPolynomial[nonZeroARI[ariIndex,1]];
                    randomParameters[randomParameters[,thetas[i]]-ariPolynomial[nonZeroARI[ariIndex,1]]>psiBounds[2],thetas[i]] <-
                        psiBounds[2]+ariPolynomial[nonZeroARI[ariIndex,1]];
                }
                else{
                    randomParameters[randomParameters[,thetas[i]]<psiBounds[1],thetas[i]] <- psiBounds[1];
                    randomParameters[randomParameters[,thetas[i]]>psiBounds[2],thetas[i]] <- psiBounds[2];
                }
            }
        }
        # Locate phi for ARIMA (they are always phi1, phi2 etc)
        if(length(phis)>0){
            # AR parameters
            for(i in 1:length(phis)){
                # Get bounds for AR based on stationarity condition
                phiBounds <- arPolinomialsBounds(arPolynomialMatrix, arPolynomial,
                                                 which(arPolynomial==arPolynomial[arPolynomial!=0][-1][i]));

                randomParameters[randomParameters[,phis[i]]<phiBounds[1],phis[i]] <- phiBounds[1];
                randomParameters[randomParameters[,phis[i]]>phiBounds[2],phis[i]] <- phiBounds[2];
            }
        }
    }

    # Set the bounds for deltas
    if(any(substr(parametersNames,1,5)=="delta")){
        deltas <- which(substr(colnames(randomParameters),1,5)=="delta");
        randomParameters[,deltas][randomParameters[,deltas]<0] <- 0;
        randomParameters[,deltas][randomParameters[,deltas]>1] <- 1;
    }

    #### Prepare the necessary matrices ####
    # States are defined similar to how it is done in adam.
    # Inserting the existing one is needed in order to deal with the case, when one of the initials was provided
    arrVt <- array(t(object$states),c(ncol(object$states),nrow(object$states),nsim),
                   dimnames=list(colnames(object$states),NULL,paste0("nsim",c(1:nsim))));
    # Set the proper time stamps for the fitted
    if(any(yClasses=="zoo")){
        fittedMatrix <- zoo(array(NA,c(obsInSample,nsim),
                                  dimnames=list(NULL,paste0("nsim",c(1:nsim)))),
                            order.by=time(yInSample));
    }
    else{
        fittedMatrix <- ts(array(NA,c(obsInSample,nsim),
                                 dimnames=list(NULL,paste0("nsim",c(1:nsim)))),
                           start=start(yInSample), frequency=frequency(yInSample));
    }

    # Transition and measurement
    arrF <- array(object$transition,c(dim(object$transition),nsim));
    arrWt <- array(object$measurement,c(dim(object$measurement),nsim));

    # Persistence matrix
    if(xregModel && !any(substr(parametersNames,1,5)=="delta")){
        matG <- array(c(object$persistence,rep(0,xregNumber)), c(length(object$persistence)+xregNumber, nsim),
                      dimnames=list(c(names(object$persistence),paste0("delta",c(1:xregNumber))),
                                    paste0("nsim",c(1:nsim))));
    }
    else{
        matG <- array(object$persistence, c(length(object$persistence), nsim),
                      dimnames=list(names(object$persistence), paste0("nsim",c(1:nsim))));
    }

    #### Fill in the values in matrices ####
    # k is the index for randomParameters columns
    k <- 0;
    # Fill in the persistence
    if(etsModel){
        if(any(parametersNames=="alpha")){
            matG["alpha",] <- randomParameters[,"alpha"];
            k <- k+1;
        }
        if(any(parametersNames=="beta")){
            matG["beta",] <- randomParameters[,"beta"];
            k <- k+1;
        }
        if(any(substr(parametersNames,1,5)=="gamma")){
            gammas <- which(substr(colnames(randomParameters),1,5)=="gamma");
            matG[colnames(randomParameters)[gammas],] <- t(randomParameters[,gammas,drop=FALSE]);
            k <- k+length(gammas);
        }

        # If we have phi, update the transition and measurement matrices
        if(any(parametersNames=="phi")){
            arrF[1,2,] <- arrF[2,2,] <- randomParameters[,"phi"];
            arrWt[,2,] <- matrix(randomParameters[,"phi"],nrow(object$measurement),nsim,byrow=TRUE);
            k <- k+1;
        }
    }
    if(xregModel && any(substr(parametersNames,1,5)=="delta")){
        deltas <- which(substr(colnames(randomParameters),1,5)=="delta");
        matG[colnames(randomParameters)[deltas],] <- t(randomParameters[,deltas,drop=FALSE]);
        k <- k+length(deltas);
    }
    # Fill in the persistence and transition for ARIMA
    if(arimaModel){
        if(is.list(object$orders)){
            arOrders <- object$orders$ar;
            iOrders <- object$orders$i;
            maOrders <- object$orders$ma;
        }
        else if(is.vector(object$orders)){
            arOrders <- object$orders[1];
            iOrders <- object$orders[2];
            maOrders <- object$orders[3];
        }

        # See if AR is needed
        arRequired <- FALSE;
        if(sum(arOrders)>0){
            arRequired[] <- TRUE;
        }
        # See if I is needed
        iRequired <- FALSE;
        if(sum(iOrders)>0){
            iRequired[] <- TRUE;
        }
        # See if I is needed
        maRequired <- FALSE;
        if(sum(maOrders)>0){
            maRequired[] <- TRUE;
        }

        # Define maxOrder and make all the values look similar (for the polynomials)
        maxOrder <- max(length(arOrders),length(iOrders),length(maOrders),length(lags));
        if(length(arOrders)!=maxOrder){
            arOrders <- c(arOrders,rep(0,maxOrder-length(arOrders)));
        }
        if(length(iOrders)!=maxOrder){
            iOrders <- c(iOrders,rep(0,maxOrder-length(iOrders)));
        }
        if(length(maOrders)!=maxOrder){
            maOrders <- c(maOrders,rep(0,maxOrder-length(maOrders)));
        }
        if(length(lags)!=maxOrder){
            lagsNew <- c(lags,rep(0,maxOrder-length(lags)));
            arOrders <- arOrders[lagsNew!=0];
            iOrders <- iOrders[lagsNew!=0];
            maOrders <- maOrders[lagsNew!=0];
        }
        # The provided parameters
        armaParameters <- object$other$armaParameters;
        # Check if the AR / MA parameters were estimated
        arEstimate <- any((substr(parametersNames,1,3)=="phi") & (nchar(parametersNames)>3))
        maEstimate <- any(substr(parametersNames,1,5)=="theta");

        # polyIndex is the index of the phi / theta parameters -1
        if(any(c(arEstimate,maEstimate))){
            polyIndex <- min(which((substr(parametersNames,1,3)=="phi") & (nchar(parametersNames)>3)),
                             which(substr(parametersNames,1,5)=="theta")) -1;
        }
        # If AR / MA are not estimated, then we don't care
        else{
            polyIndex <- -1;
        }

        for(i in 1:nsim){
            # Call the function returning ARI and MA polynomials
            arimaPolynomials <- polynomialiser(randomParameters[i,polyIndex+1:sum(c(arOrders*arEstimate,maOrders*maEstimate))],
                                               arOrders, iOrders, maOrders, arRequired, maRequired, arEstimate, maEstimate,
                                               armaParameters, lags);

            # Fill in the transition and persistence matrices
            if(nrow(nonZeroARI)>0){
                arrF[componentsNumberETS+nonZeroARI[,2],componentsNumberETS+1:componentsNumberARIMA,i] <-
                    -arimaPolynomials$ariPolynomial[nonZeroARI[,1]];
                matG[componentsNumberETS+nonZeroARI[,2],i] <- -arimaPolynomials$ariPolynomial[nonZeroARI[,1]];
            }
            if(nrow(nonZeroMA)>0){
                matG[componentsNumberETS+nonZeroMA[,2],i] <- matG[componentsNumberETS+nonZeroMA[,2],i] +
                    arimaPolynomials$maPolynomial[nonZeroMA[,1]];
            }
        }
        k <- k+sum(c(arOrders*arEstimate,maOrders*maEstimate));
    }

    # j is the index for the components in the profile
    j <- 0
    # Fill in the profile values
    profilesRecentArray <- array(t(object$states[1:lagsModelMax,]),c(dim(object$profile),nsim));
    if(etsModel && object$initialType=="optimal"){
        if(any(parametersNames=="level")){
            j <- j+1;
            profilesRecentArray[j,1,] <- randomParameters[,"level"];
            k <- k+1;
        }
        if(any(parametersNames=="trend")){
            j <- j+1;
            profilesRecentArray[j,1,] <- randomParameters[,"trend"];
            k <- k+1;
        }
        if(any(substr(parametersNames,1,8)=="seasonal")){
            # If there is only one seasonality
            if(any(substr(parametersNames,1,9)=="seasonal_")){
                initialSeasonalIndices <- 1;
                seasonalNames <- "seasonal"
            }
            # If there are several
            else{
                # This assumes that we cannot have more than 9 seasonalities.
                initialSeasonalIndices <- as.numeric(unique(substr(parametersNames[substr(parametersNames,1,8)=="seasonal"],9,9)));
                seasonalNames <- unique(substr(parametersNames[substr(parametersNames,1,8)=="seasonal"],1,9));
            }
            for(i in initialSeasonalIndices){
                profilesRecentArray[j+i,1:(lagsSeasonal[i]-1),] <-
                    t(randomParameters[,paste0(seasonalNames[i],"_",c(1:(lagsSeasonal[i]-1)))]);
                profilesRecentArray[j+i,lagsSeasonal[i],] <-
                    switch(Stype,
                           "A"=-apply(profilesRecentArray[j+i,1:(lagsSeasonal[i]-1),,drop=FALSE],3,sum),
                           "M"=1/apply(profilesRecentArray[j+i,1:(lagsSeasonal[i]-1),,drop=FALSE],3,prod),
                           0);
            }
            j <- j+max(initialSeasonalIndices);
            k <- k+length(initialSeasonalIndices);
        }
    }
    # ARIMA states in the profileRecent
    if(arimaModel){
        # See if the initials were estimated
        initialArimaNumber <- sum(substr(parametersNames,1,10)=="ARIMAState");

        # This is needed in order to propagate initials of ARIMA to all components
        if(object$initialType=="optimal" && any(c(arEstimate,maEstimate))){
            if(nrow(nonZeroARI)>0 && nrow(nonZeroARI)>=nrow(nonZeroMA)){
                for(i in 1:nsim){
                    # Call the function returning ARI and MA polynomials
                    ### This is not optimal, as the polynomialiser() is called twice (for parameters and here),
                    ### but this is simpler
                    arimaPolynomials <- polynomialiser(randomParameters[i,polyIndex+1:sum(c(arOrders*arEstimate,maOrders*maEstimate))],
                                                       arOrders, iOrders, maOrders, arRequired, maRequired, arEstimate, maEstimate,
                                                       armaParameters, lags);
                    profilesRecentArray[componentsNumberETS+componentsNumberARIMA, 1:initialArimaNumber, i] <-
                        randomParameters[i, k+1:initialArimaNumber];
                    profilesRecentArray[j+nonZeroARI[,2], 1:initialArimaNumber, i] <-
                        switch(Etype,
                               "A"= arimaPolynomials$ariPolynomial[nonZeroARI[,1]] %*%
                                   t(profilesRecentArray[componentsNumberETS+componentsNumberARIMA,
                                                         1:initialArimaNumber, i]) /
                                   tail(arimaPolynomials$ariPolynomial,1),
                               "M"=exp(arimaPolynomials$ariPolynomial[nonZeroARI[,1]] %*%
                                           t(log(profilesRecentArray[componentsNumberETS+componentsNumberARIMA,
                                                                     1:initialArimaNumber, i])) /
                                           tail(arimaPolynomials$ariPolynomial,1)));
                }
            }
            else{
                for(i in 1:nsim){
                    # Call the function returning ARI and MA polynomials
                    arimaPolynomials <- polynomialiser(randomParameters[i,polyIndex+1:sum(c(arOrders*arEstimate,maOrders*maEstimate))],
                                                       arOrders, iOrders, maOrders, arRequired, maRequired, arEstimate, maEstimate,
                                                       armaParameters, lags);
                    profilesRecentArray[componentsNumberETS+componentsNumberARIMA, 1:initialArimaNumber, i] <-
                        randomParameters[i, k+1:initialArimaNumber];
                    profilesRecentArray[j+nonZeroMA[,2], 1:initialArimaNumber, i] <-
                        switch(Etype,
                               "A"=arimaPolynomials$maPolynomial[nonZeroMA[,1]] %*%
                                   t(profilesRecentArray[componentsNumberETS+componentsNumberARIMA,
                                                         1:initialArimaNumber, i]) /
                                   tail(arimaPolynomials$maPolynomial,1),
                               "M"=exp(arimaPolynomials$maPolynomial[nonZeroMA[,1]] %*%
                                           t(log(profilesRecentArray[componentsNumberETS+componentsNumberARIMA,
                                                                     1:initialArimaNumber, i])) /
                                           tail(arimaPolynomials$maPolynomial,1)));
                }
            }
        }
        j <- j+initialArimaNumber;
        k <- k+initialArimaNumber;
    }
    if(xregModel){
        xregNumberToEstimate <- sum(xregParametersEstimated);
        profilesRecentArray[j+which(xregParametersEstimated==1),1,] <- t(randomParameters[,k+1:xregNumberToEstimate]);
        # Normalise initials
        for(i in which(xregParametersMissing!=0)){
            profilesRecentArray[j+i,1,] <- -colSums(profilesRecentArray[j+which(xregParametersEstimated==1),1,]);
        }
    }

    if(is.null(object$occurrence)){
        ot <- matrix(rep(1, obsInSample));
        pt <- rep(1, obsInSample);
    }
    else{
        ot <- matrix(actuals(object$occurrence));
        pt <- fitted(object$occurrence);
    }

    yt <- matrix(actuals(object));

    # Refit the model with the new parameter
    adamRefitted <- adamRefitterWrap(yt, ot, arrVt, arrF, arrWt, matG,
                                     Etype, Ttype, Stype,
                                     lagsModelAll, profilesObservedTable, profilesRecentArray,
                                     componentsNumberETSSeasonal, componentsNumberETS,
                                     componentsNumberARIMA, xregNumber);
    arrVt[] <- adamRefitted$states;
    fittedMatrix[] <- adamRefitted$fitted * as.vector(pt);
    profilesRecentArray[] <- adamRefitted$profilesRecent;

    return(structure(list(y=actuals(object), states=arrVt, refitted=fittedMatrix,
                          fitted=fitted(object), model=object$model,
                          timeElapsed=Sys.time()-startTime,
                          transition=arrF, measurement=arrWt, persistence=matG,
                          profile=profilesRecentArray),
                     class="refit"));
}

#' @importFrom grDevices rgb
#' @export
plot.refit <- function(x, ...){
    ellipsis <- list(...);
    ellipsis$x <- actuals(x);
    nsim <- ncol(x$refitted);

    if(any(class(ellipsis$x)=="zoo")){
        yQuantiles <- zoo(matrix(0,length(ellipsis$x),11),order.by=time(ellipsis$x));
    }
    else{
        yQuantiles <- ts(matrix(0,length(ellipsis$x),11),start=start(ellipsis$x),frequency=frequency(ellipsis$x));
    }
    quantileseq <- seq(0,1,length.out=11);
    yQuantiles[,1] <- apply(x$refitted,1,quantile,0.975,na.rm=TRUE);
    yQuantiles[,11] <- apply(x$refitted,1,quantile,0.025,na.rm=TRUE);
    for(i in 2:10){
        yQuantiles[,i] <- apply(x$refitted,1,quantile,quantileseq[i],na.rm=TRUE);
    }

    if(is.null(ellipsis$ylim)){
        ellipsis$ylim <- range(c(ellipsis$x,fitted(x)),na.rm=TRUE);
    }
    if(is.null(ellipsis$main)){
        ellipsis$main <- paste0("Refitted values of ",x$model);
    }
    if(is.null(ellipsis$ylab)){
        ellipsis$ylab <- "";
    }

    do.call(plot, ellipsis);
    polygon(c(time(yQuantiles),rev(time(yQuantiles))), c(yQuantiles[,1],rev(yQuantiles[,11])), col=rgb(0.8,0.8,0.8,0.4), border="grey")
    polygon(c(time(yQuantiles),rev(time(yQuantiles))), c(yQuantiles[,2],rev(yQuantiles[,10])), col=rgb(0.8,0.8,0.8,0.5), border="grey")
    polygon(c(time(yQuantiles),rev(time(yQuantiles))), c(yQuantiles[,3],rev(yQuantiles[,9])), col=rgb(0.8,0.8,0.8,0.6), border="grey")
    polygon(c(time(yQuantiles),rev(time(yQuantiles))), c(yQuantiles[,4],rev(yQuantiles[,8])), col=rgb(0.8,0.8,0.8,0.7), border="grey")
    polygon(c(time(yQuantiles),rev(time(yQuantiles))), c(yQuantiles[,5],rev(yQuantiles[,7])), col=rgb(0.8,0.8,0.8,0.8), border="grey")
    lines(ellipsis$x,col="black",lwd=1);
    lines(fitted(x),col="purple",lwd=2,lty=2);
}

#' @export
print.refit <- function(x, ...){
    nsim <- ncol(x$refitted);
    cat("Time elapsed:",round(as.numeric(x$timeElapsed,units="secs"),2),"seconds");
    cat("\nModel refitted:",x$model);
    cat("\nNumber of simulation paths produced:",nsim);
}

#' @rdname refit
#' @export reforecast
reforecast <- function(object, h=10, newdata=NULL, occurrence=NULL,
                       interval=c("prediction", "confidence", "none"),
                       level=0.95, side=c("both","upper","lower"), cumulative=FALSE,
                       nsim=100, ...) UseMethod("reforecast")

#' @export
reforecast.default <- function(object, h=10, newdata=NULL, occurrence=NULL,
                               interval=c("prediction", "confidence", "none"),
                               level=0.95, side=c("both","upper","lower"), cumulative=FALSE,
                               nsim=100, ...){
    warning(paste0("The method is not implemented for the object of the class ,",class(object)[1]),
            call.=FALSE);
    return(forecast(object=object, h=h, newdata=newdata, occurrence=occurrence,
                    interval=interval, level=level, side=side, cumulative=cumulative,
                    nsim=nsim, ...));
}

#' @export
reforecast.adam <- function(object, h=10, newdata=NULL, occurrence=NULL,
                            interval=c("prediction", "confidence", "none"),
                            level=0.95, side=c("both","upper","lower"), cumulative=FALSE,
                            nsim=100, ...){
    objectRefitted <- refit(object, nsim=nsim, ...);
    ellipsis <- list(...);

    # If the trim is not provided, set it to 1%
    if(is.null(ellipsis$trim)){
        trim <- 0.01;
    }
    else{
        trim <- ellipsis$trim;
    }

    #### <--- This part is widely a copy-paste from forecast.adam()
    interval <- match.arg(interval[1],c("none", "prediction", "confidence","simulated"));
    side <- match.arg(side);

    # Model type
    model <- modelType(object);
    Etype <- errorType(object);
    Ttype <- substr(model,2,2);
    Stype <- substr(model,nchar(model),nchar(model));

    # Technical parameters
    lagsModelAll <- modelLags(object);
    lagsModelMax <- max(lagsModelAll);
    profilesRecentArray <- objectRefitted$profile;

    if(!is.null(object$initial$seasonal)){
        if(is.list(object$initial$seasonal)){
            componentsNumberETSSeasonal <- length(object$initial$seasonal);
        }
        else{
            componentsNumberETSSeasonal <- 1;
        }
    }
    else{
        componentsNumberETSSeasonal <- 0;
    }
    componentsNumberETS <- length(object$initial$level) + length(object$initial$trend) + componentsNumberETSSeasonal;
    componentsNumberARIMA <- sum(substr(colnames(object$states),1,10)=="ARIMAState");

    obsStates <- nrow(object$states);
    obsInSample <- nobs(object);
    profilesObservedTable <- adamProfileCreator(lagsModelAll, lagsModelMax, obsInSample+h)$observed[,-c(1:obsInSample),drop=FALSE];

    yClasses <- class(actuals(object));

    if(any(yClasses=="ts")){
        # ts structure
        if(h>0){
            yForecastStart <- time(actuals(object))[obsInSample]+deltat(actuals(object));
        }
        else{
            yForecastStart <- time(actuals(object))[1];
        }
        yFrequency <- frequency(actuals(object));
    }
    else{
        # zoo thingy
        yIndex <- time(actuals(object));
        if(h>0){
            yForecastIndex <- yIndex[obsInSample]+diff(tail(yIndex,2))*c(1:h);
        }
        else{
            yForecastIndex <- yIndex;
        }
    }

    # How many levels did user asked to produce
    nLevels <- length(level);
    # Cumulative forecasts have only one observation
    if(cumulative){
        # hFinal is the number of elements we will have in the final forecast
        hFinal <- 1;
    }
    else{
        if(h>0){
            hFinal <- h;
        }
        else{
            hFinal <- obsInSample;
        }
    }

    # Create necessary matrices for the forecasts
    if(any(yClasses=="ts")){
        yForecast <- ts(vector("numeric", hFinal), start=yForecastStart, frequency=yFrequency);
        yUpper <- yLower <- ts(matrix(0,hFinal,nLevels), start=yForecastStart, frequency=yFrequency);
    }
    else{
        yForecast <- zoo(vector("numeric", hFinal), order.by=yForecastIndex);
        yUpper <- yLower <- zoo(matrix(0,hFinal,nLevels), order.by=yForecastIndex);
    }

    # If the occurrence values are provided for the holdout
    if(!is.null(occurrence) && is.numeric(occurrence)){
        pForecast <- occurrence;
    }
    else{
        # If this is a mixture model, produce forecasts for the occurrence
        if(is.occurrence(object$occurrence)){
            occurrenceModel <- TRUE;
            if(is.alm(object$occurrence)){
                pForecast <- suppressWarnings(forecast(object$occurrence,h=h,newdata=newdata)$mean);
            }
            else{
                pForecast <- suppressWarnings(forecast(object$occurrence,h=h,newdata=newdata)$mean);
            }
        }
        else{
            occurrenceModel <- FALSE;
            # If this was provided occurrence, then use provided values
            if(!is.null(object$occurrence) && !is.null(object$occurrence$occurrence) &&
               (object$occurrence$occurrence=="provided")){
                pForecast <- object$occurrence$forecast;
            }
            else{
                pForecast <- rep(1, h);
            }
        }
    }

    # Make sure that the values are of the correct length
    if(h<length(pForecast)){
        pForecast <- pForecast[1:h];
    }
    else if(h>length(pForecast)){
        pForecast <- c(pForecast, rep(tail(pForecast,1), h-length(pForecast)));
    }

    # Set the levels
    if(interval!="none"){
        # Fix just in case a silly user used 95 etc instead of 0.95
        if(any(level>1)){
            level[] <- level / 100;
        }
        levelLow <- levelUp <- matrix(0,hFinal,nLevels);
        levelNew <- matrix(level,nrow=hFinal,ncol=nLevels,byrow=TRUE);

        # If this is an occurrence model, then take probability into account in the level.
        # This correction is only needed for approximate, because the others contain zeroes
        if(side=="both"){
            levelLow[] <- (1-levelNew)/2;
            levelUp[] <- (1+levelNew)/2;
        }
        else if(side=="upper"){
            levelLow[] <- 0;
            levelUp[] <- levelNew;
        }
        else{
            levelLow[] <- 1-levelNew;
            levelUp[] <- 1;
        }
        levelLow[levelLow<0] <- 0;
        levelUp[levelUp<0] <- 0;
    }

    #### Return adam.predict if h<=0 ####
    # If the horizon is zero, just construct fitted and potentially confidence interval thingy
    if(h<=0){
        # If prediction interval is needed, this can be done with predict.adam
        if(any(interval==c("prediction","none"))){
            warning(paste0("You've set h=",h," and interval=\"",interval,
                           "\". There is no point in using reforecast() function for your task. ",
                           "Using predict() method instead."),
                    call.=FALSE);
            return(predict(object, newdata=newdata,
                           interval=interval,
                           level=level, side=side, ...));
        }

        yForecast[] <- rowMeans(objectRefitted$refitted);
        if(interval=="confidence"){
            for(i in 1:hFinal){
                yLower[i,] <- quantile(objectRefitted$refitted[i,],levelLow[i,]);
                yUpper[i,] <- quantile(objectRefitted$refitted[i,],levelUp[i,]);
            }
        }
        return(structure(list(mean=yForecast, lower=yLower, upper=yUpper, model=object,
                              level=level, interval=interval, side=side),
                         class=c("adam.predict","adam.forecast")));
    }

    #### All the important matrices ####
    # Last h observations of measurement
    arrWt <- objectRefitted$measurement[obsInSample-c(h:1)+1,,,drop=FALSE];
    # If the forecast horizon is higher than the in-sample, duplicate the last value in matWt
    if(dim(arrWt)[1]<h){
        arrWt <- array(tail(arrWt,1), c(h, ncol(arrWt), nsim), dimnames=list(NULL,colnames(arrWt),NULL));
    }

    # Deal with explanatory variables
    if(ncol(object$data)>1){
        xregNumber <- length(object$initial$xreg);
        xregNames <- names(object$initial$xreg);
        if(is.null(newdata) && !is.null(object$holdout) && nrow(object$holdout)<h){
            xreg <- tail(object$data,h);
            if(!is.data.frame(newdata) & is.matrix(xreg)){
                warning("The newdata is not provided.",
                        "Predicting the explanatory variables based on what we have in-sample.",
                        call.=FALSE);
                for(i in 1:xregNumber){
                    xreg[,i] <- adam(object$data[,i+1],h=h,silent=TRUE)$forecast;
                }
            }
            else{
                warning("The newdata is not provided. Using last h in-sample observations instead.",
                        call.=FALSE);
            }
        }
        else if(is.null(newdata) && !is.null(object$holdout) && nrow(object$holdout)>=h){
            xreg <- object$holdout[1:h,,drop=FALSE];
        }
        else{
            # If this is not a matrix / data.frame, then convert to one
            if(!is.data.frame(newdata) && !is.matrix(newdata)){
                newdata <- as.data.frame(newdata);
                colnames(newdata) <- "xreg";
            }
            if(nrow(newdata)<h){
                warning(paste0("The newdata has ",nrow(newdata)," observations, while ",h," are needed. ",
                               "Using the last available values as future ones."),
                        call.=FALSE);
                newnRows <- h-nrow(newdata);
                xreg <- rbind(newdata,matrix(rep(tail(newdata,1),each=newnRows),newnRows,ncol(newdata)));
            }
            else if(nrow(newdata)>h){
                warning(paste0("The newdata has ",nrow(newdata)," observations, while only ",h," are needed. ",
                               "Using the last ",h," of them."),
                        call.=FALSE);
                xreg <- tail(newdata,h);
            }
            else{
                xreg <- newdata;
            }
        }

        # If the names are wrong, transform to data frame and expand
        if(!all(xregNames %in% colnames(xreg))){
            xreg <- as.data.frame(xreg);
        }

        # Expand the xreg if it is data frame to get the proper matrix
        if(is.data.frame(xreg)){
            testFormula <- formula(object);
            # Expand the variables. We cannot use alm, because it is based on obsInSample
            xregData <- model.frame(testFormula,data=xreg);
            # Binary, flagging factors in the data
            # Expanded stuff with all levels for factors
            if(any((attr(terms(xregData),"dataClasses")=="factor")[-1])){
                xregModelMatrix <- model.matrix(xregData,xregData,
                                                contrasts.arg=lapply(xregData[attr(terms(xregData),"dataClasses")=="factor"],
                                                                     contrasts, contrasts=FALSE));
            }
            else{
                xregModelMatrix <- model.matrix(xregData,data=xregData);
            }
            colnames(xregModelMatrix) <- make.names(colnames(xregModelMatrix), unique=TRUE);
            newdata <- as.matrix(xregModelMatrix)[,xregNames,drop=FALSE];
            rm(xregData,xregModelMatrix);
        }
        else{
            newdata <- xreg[,xregNames];
        }
        rm(xreg);

        arrWt[,componentsNumberETS+componentsNumberARIMA+c(1:xregNumber),] <- newdata;
    }
    else{
        xregNumber <- 0;
    }

    #### Simulate the data ####
    sigmaValue <- sigma(object);
    # This stuff is needed in order to produce adequate values for weird models
    EtypeModified <- Etype;
    if(Etype=="A" && any(object$distribution==c("dlnorm","dinvgauss","dls","dllaplace"))){
        EtypeModified[] <- "M";
    }
    # Matrix for the errors
    arrErrors <- array(switch(object$distribution,
                              "dnorm"=rnorm(h*nsim^2, 0, sigmaValue),
                              "dlaplace"=rlaplace(h*nsim^2, 0, sigmaValue/2),
                              "ds"=rs(h*nsim^2, 0, (sigmaValue^2/120)^0.25),
                              "dgnorm"=rgnorm(h*nsim^2, 0,
                                              sigmaValue*sqrt(gamma(1/object$other$beta)/gamma(3/object$other$beta))),
                              "dlogis"=rlogis(h*nsim^2, 0, sigmaValue*sqrt(3)/pi),
                              "dt"=rt(h*nsim^2, obsInSample-nparam(object)),
                              "dalaplace"=ralaplace(h*nsim^2, 0,
                                                    sqrt(sigmaValue^2*object$other$alpha^2*(1-object$other$alpha)^2/
                                                             (object$other$alpha^2+(1-object$other$alpha)^2)),
                                                    object$other$alpha),
                              "dlnorm"=rlnorm(h*nsim^2, -object$scale^2/2, object$scale)-1,
                              "dinvgauss"=rinvgauss(h*nsim^2, 1, dispersion=sigmaValue^2)-1,
                              "dllaplace"=exp(rlaplace(h*nsim^2, 0, sigmaValue/2))-1,
                              "dls"=exp(rs(h*nsim^2, 0, (sigmaValue^2/120)^0.25))-1,
                              "dlgnorm"=exp(rgnorm(h*nsim^2, 0,
                                                   sigmaValue*sqrt(gamma(1/object$other$beta)/gamma(3/object$other$beta))))-1),
                       c(h,nsim,nsim));
    # Normalise errors in order not to get ridiculous things on small nsim
    if(nsim<=500){
        if(Etype=="A"){
            arrErrors[] <- arrErrors - array(apply(arrErrors,1,mean),c(h,nsim,nsim));
        }
        else{
            arrErrors[] <- (1+arrErrors) / array(apply(1+arrErrors,1,mean),c(h,nsim,nsim))-1;
        }
    }
    # Array of the simulated data
    arrayYSimulated <- array(0,c(h,nsim,nsim));
    # Start the loop... might take some time
    arrayYSimulated[] <- adamReforecasterWrap(arrErrors,
                                              array(rbinom(h*nsim^2, 1, pForecast), c(h,nsim,nsim)),
                                              objectRefitted$transition,
                                              arrWt,
                                              objectRefitted$persistence,
                                              EtypeModified, Ttype, Stype,
                                              lagsModelAll, profilesObservedTable, profilesRecentArray,
                                              componentsNumberETSSeasonal, componentsNumberETS,
                                              componentsNumberARIMA, xregNumber)$matrixYt;

    #### Note that the cumulative doesn't work with oes at the moment!
    if(cumulative){
        yForecast[] <- mean(apply(arrayYSimulated,1,sum,na.rm=T,trim=trim));
        if(interval!="none"){
            yLower[] <- quantile(apply(arrayYSimulated,1,sum,na.rm=T),levelLow,type=7);
            yUpper[] <- quantile(apply(arrayYSimulated,1,sum,na.rm=T),levelUp,type=7);
        }
    }
    else{
        yForecast[] <- apply(arrayYSimulated,1,mean,na.rm=T,trim=trim);
        if(interval=="prediction"){
            for(i in 1:h){
                for(j in 1:nLevels){
                    yLower[i,j] <- quantile(arrayYSimulated[i,,],levelLow[i,j],na.rm=T,type=7);
                    yUpper[i,j] <- quantile(arrayYSimulated[i,,],levelUp[i,j],na.rm=T,type=7);
                }
            }
        }
        else if(interval=="confidence"){
            for(i in 1:h){
                yLower[i,] <- quantile(apply(arrayYSimulated[i,,],2,mean,na.rm=T,trim=trim),levelLow[i,],na.rm=T,type=7);
                yUpper[i,] <- quantile(apply(arrayYSimulated[i,,],2,mean,na.rm=T,trim=trim),levelUp[i,],na.rm=T,type=7);
            }
        }
    }

    # Fix of prediction intervals depending on what has happened
    if(interval!="none"){
        # Make sensible values out of those weird quantiles
        if(!cumulative){
            if(any(levelLow==0)){
                # zoo does not like, when you work with matrices of indices... silly thing
                yBoundBuffer <- levelLow;
                yBoundBuffer[] <- yLower
                if(Etype=="A"){
                    yBoundBuffer[levelLow==0] <- -Inf;
                    yLower[] <- yBoundBuffer;
                }
                else{
                    yBoundBuffer[levelLow==0] <- 0;
                    yLower[] <- yBoundBuffer;
                }
            }
            if(any(levelUp==1)){
                # zoo does not like, when you work with matrices of indices... silly thing
                yBoundBuffer <- levelUp;
                yBoundBuffer[] <- yUpper
                yBoundBuffer[levelUp==1] <- Inf;
                yUpper[] <- yBoundBuffer;
            }
        }
        else{
            if(Etype=="A" && any(levelLow==0)){
                yLower[] <- -Inf;
            }
            else if(Etype=="M" && any(levelLow==0)){
                yLower[] <- 0;
            }
            if(any(levelUp==1)){
                yUpper[] <- Inf;
            }
        }

        # Substitute NAs and NaNs with zeroes
        if(any(is.nan(yLower)) || any(is.na(yLower))){
            yLower[is.nan(yLower)] <- switch(Etype,"A"=0,"M"=1);
            yLower[is.na(yLower)] <- switch(Etype,"A"=0,"M"=1);
        }
        if(any(is.nan(yUpper)) || any(is.na(yUpper))){
            yUpper[is.nan(yUpper)] <- switch(Etype,"A"=0,"M"=1);
            yUpper[is.na(yUpper)] <- switch(Etype,"A"=0,"M"=1);
        }

        # Check what we have from the occurrence model
        if(occurrenceModel){
            # If there are NAs, then there's no variability and no intervals.
            if(any(is.na(yUpper))){
                yUpper[is.na(yUpper)] <- (yForecast/pForecast)[is.na(yUpper)];
            }
            if(any(is.na(yLower))){
                yLower[is.na(yLower)] <- 0;
            }
        }

        colnames(yLower) <- switch(side,
                                   "both"=paste0("Lower bound (",(1-level)/2*100,"%)"),
                                   "lower"=paste0("Lower bound (",(1-level)*100,"%)"),
                                   "upper"=rep("Lower 0%",nLevels));

        colnames(yUpper) <- switch(side,
                                   "both"=paste0("Upper bound (",(1+level)/2*100,"%)"),
                                   "lower"=rep("Upper 100%",nLevels),
                                   "upper"=paste0("Upper bound (",level*100,"%)"));
    }
    else{
        yUpper[] <- yLower[] <- NA;
    }

    structure(list(mean=yForecast, lower=yLower, upper=yUpper, model=object,
                   level=level, interval=interval, side=side, cumulative=cumulative,
                   paths=arrayYSimulated),
              class=c("adam.forecast","smooth.forecast","forecast"));
}


#### Other methods ####

#' @export
multicov.adam <- function(object, type=c("analytical","empirical","simulated"), ...){
    type <- match.arg(type);

    # Model type
    Ttype <- substr(modelType(object),2,2);

    h <- length(object$holdout);
    lagsModelAll <- modelLags(object);
    lagsModelMax <- max(lagsModelAll);
    lagsOriginal <- lags(object);
    if(Ttype!="N"){
        lagsOriginal <- c(1,lagsOriginal);
    }
    componentsNumberETS <- length(lagsOriginal);
    componentsNumberETSSeasonal <- sum(lagsOriginal>1);
    componentsNumberARIMA <- sum(substr(colnames(object$states),1,10)=="ARIMAState");

    s2 <- sigma(object)^2;
    matWt <- tail(object$measurement,h);
    vecG <- matrix(object$persistence, ncol=1);
    if(ncol(object$data)>1){
        xregNumber <- ncol(object$data)-1;
    }
    else{
        xregNumber <- 0;
    }
    matF <- object$transition;

    if(type=="analytical"){
        covarMat <- covarAnal(lagsModelAll, h, matWt[1,,drop=FALSE], matF, vecG, s2);
    }
    else if(type=="empirical"){
        adamErrors <- rmultistep(object, h=h);
        covarMat <- t(adamErrors) %*% adamErrors / (nobs(object) - h);
    }

    return(covarMat);
}

#' @export
pointLik.adam <- function(object, ...){
    distribution <- object$distribution;
    yInSample <- actuals(object);
    obsInSample <- nobs(object);
    if(is.occurrence(object$occurrence)){
        otLogical <- yInSample!=0;
        yFitted <- fitted(object) / fitted(object$occurrence);
    }
    else{
        otLogical <- rep(TRUE, obsInSample);
        yFitted <- fitted(object);
    }
    scale <- object$scale;
    other <- switch(distribution,
                    "dalaplace"=object$other$alpha,
                    "dgnorm"=,"dlgnorm"=object$other$beta,
                    "dt"=object$other$nu);
    Etype <- errorType(object);

    likValues <- vector("numeric",obsInSample);
    likValues[otLogical] <- switch(distribution,
                                   "dnorm"=switch(Etype,
                                                  "A"=dnorm(x=yInSample[otLogical], mean=yFitted[otLogical],
                                                            sd=scale, log=TRUE),
                                                  "M"=dnorm(x=yInSample[otLogical], mean=yFitted[otLogical],
                                                            sd=scale*yFitted[otLogical], log=TRUE)),
                                   "dlaplace"=switch(Etype,
                                                     "A"=dlaplace(q=yInSample[otLogical], mu=yFitted[otLogical],
                                                                  scale=scale, log=TRUE),
                                                     "M"=dlaplace(q=yInSample[otLogical], mu=yFitted[otLogical],
                                                                  scale=scale*yFitted[otLogical], log=TRUE)),
                                   "ds"=switch(Etype,
                                               "A"=ds(q=yInSample[otLogical],mu=yFitted[otLogical],
                                                      scale=scale, log=TRUE),
                                               "M"=ds(q=yInSample[otLogical],mu=yFitted[otLogical],
                                                      scale=scale*sqrt(yFitted[otLogical]), log=TRUE)),
                                   "dgnorm"=switch(Etype,
                                                  "A"=dgnorm(x=yInSample[otLogical], mu=yFitted[otLogical],
                                                            alpha=scale, beta=other, log=TRUE),
                                                  "M"=suppressWarnings(dgnorm(x=yInSample[otLogical], mu=yFitted[otLogical],
                                                                              alpha=scale*yFitted[otLogical], beta=other,
                                                                              log=TRUE))),
                                   "dlogis"=switch(Etype,
                                                   "A"=dlogis(x=yInSample[otLogical], location=yFitted[otLogical],
                                                              scale=scale, log=TRUE),
                                                   "M"=dlogis(x=yInSample[otLogical], location=yFitted[otLogical],
                                                              scale=scale*yFitted[otLogical], log=TRUE)),
                                   "dt"=switch(Etype,
                                               "A"=dt(adamFitted$errors[otLogical], df=abs(other), log=TRUE),
                                               "M"=dt(adamFitted$errors[otLogical]*yFitted[otLogical],
                                                      df=abs(other), log=TRUE)),
                                   "dalaplace"=switch(Etype,
                                                      "A"=dalaplace(q=yInSample[otLogical], mu=yFitted[otLogical],
                                                                    scale=scale, alpha=other, log=TRUE),
                                                      "M"=dalaplace(q=yInSample[otLogical], mu=yFitted[otLogical],
                                                                    scale=scale*yFitted[otLogical], alpha=other, log=TRUE)),
                                   "dlnorm"=dlnorm(x=yInSample[otLogical],
                                                   meanlog=log(yFitted[otLogical]) -scale^2/2,
                                                   sdlog=scale, log=TRUE),
                                   "dllaplace"=dlaplace(q=log(yInSample[otLogical]), mu=log(yFitted[otLogical]),
                                                        scale=scale, log=TRUE),
                                   "dls"=ds(q=log(yInSample[otLogical]), mu=log(yFitted[otLogical]),
                                            scale=scale, log=TRUE),
                                   "dgnorm"=dgnorm(x=log(yInSample[otLogical]), mu=log(yFitted[otLogical]),
                                                   alpha=scale, beta=other, log=TRUE),
                                   "dinvgauss"=dinvgauss(x=yInSample[otLogical], mean=yFitted[otLogical],
                                                         dispersion=scale/yFitted[otLogical], log=TRUE));
    if(any(distribution==c("dllaplace","dls","dlgnorm"))){
        likValues[otLogical] <- likValues[otLogical] - log(yInSample[otLogical]);
    }

    # If this is a mixture model, take the respective probabilities into account (differential entropy)
    if(is.occurrence(object$occurrence)){
        likValues[!otLogical] <- -switch(distribution,
                                         "dnorm" = (log(sqrt(2*pi)*scale)+0.5),
                                         "dlnorm" = (log(sqrt(2*pi)*scale)+0.5) -scale^2/2,
                                         "dlogis" = 2,
                                         "dlaplace" =,
                                         "dllaplace" =,
                                         "dalaplace" = (1 + log(2*scale)),
                                         "dt" = ((scale+1)/2 * (digamma((scale+1)/2)-digamma(scale/2)) +
                                                     log(sqrt(scale) * beta(scale/2,0.5))),
                                         "ds" =,
                                         "dls" = (2 + 2*log(2*scale)),
                                         "dgnorm" =,
                                         "dlgnorm" = 1/other-log(other/(2*scale*gamma(1/other))),
                                         "dinvgauss" = (0.5*(log(pi/2)+1+log(scale))));

        likValues[] <- likValues + pointLik(object$occurrence);
    }
    likValues <- ts(likValues, start=start(yFitted), frequency=frequency(yFitted));

    return(likValues);
}

#' @export
modelType.adam <- function(object, ...){
    etsModel <- any(unlist(gregexpr("ETS",object$model))!=-1);
    if(etsModel){
        modelType <- substring(object$model,
                               unlist(gregexpr("\\(",object$model))+1,
                               unlist(gregexpr("\\)",object$model))-1)[1];
    }
    else{
        modelType <- "NNN";
    }
    return(modelType)
}

#' @export
errorType.adam <- function(object, ...){
    model <- modelType(object);
    if(model=="NNN"){
        return(switch(object$distribution,
                      "dnorm"=,"dlaplace"=,"ds"=,"dgnorm"=,"dlogis"=,"dt"=,"dalaplace"="A",
                      "dlnorm"=,"dllaplace"=,"dls"=,"dlgnorm"=,"dinvgauss"="M"));
    }
    else{
        return(substr(model,1,1));
    }
}

# This is an internal function, no need to export it
# modelLags <- function(object, ...) UseMethod("modelLags")
modelLags.adam <- function(object, ...){
    return(object$lagsAll);
}

#' @export
orders.adam <- function(object, ...){
    return(object$orders);
}

##### Other methods to implement #####
# accuracy.adam <- function(object, holdout, ...){}
# simulate.adam <- function(object, nsim=1, seed=NULL, obs=NULL, ...){}
