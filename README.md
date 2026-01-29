# smooth

R:

[![CRAN_Status_Badge](https://www.r-pkg.org/badges/version/smooth)](https://cran.r-project.org/package=smooth)
[![Downloads](https://cranlogs.r-pkg.org/badges/smooth)](https://cran.r-project.org/package=smooth)
[![R-CMD-check](https://github.com/config-i1/smooth/actions/workflows/test.yml/badge.svg)](https://github.com/config-i1/smooth/actions/workflows/test.yml)

Python:

[![PyPI - Downloads](https://img.shields.io/pypi/dm/smooth.svg)](https://pypi.org/project/smooth/)

The package _smooth_ contains functions implementing Single Source of Error state space models for purposes of time series analysis and forecasting

![hex-sticker of the smooth package for R](https://github.com/config-i1/smooth/blob/master/man/figures/smooth-web.png?raw=true)

Here is the list of the included functions:

## Models and main functions

### Main forecasting functions

1. **adam** - Advanced Dynamic Adaptive Model, the main unified framework implementing ETS (Error-Trend-Seasonal), ARIMA (AutoRegressive Integrated Moving Average), and regression with exogenous variables. Supports multiple seasonalities, various error distributions (Normal, Laplace, S, Generalised Normal, Log-Normal, Inverse Gaussian, Gamma), and different loss functions. This is the recommended function for most forecasting tasks.
2. **auto.adam** - Automatic model selection for ADAM. Selects the best distribution, ARIMA orders (if requested), and can detect and handle outliers. Supports parallel computation for faster estimation when testing multiple distributions.
3. **es** - Exponential Smoothing (ETS) function. Implements all 30 ETS model types with automatic model selection using branch and bound algorithm. Supports exogenous variables (ETX), holdout validation, multiple loss functions including trace forecast based ones, and AIC-weighted forecast combinations. Acts as a wrapper of **adam()**.
4. **msarima** - Multiple Seasonal ARIMA. Extends ssarima to handle multiple seasonal patterns (e.g., hourly data with daily, weekly, and annual seasonality). More computationally efficient than ssarima for complex seasonal structures. Acts as a wrapper of **adam()**.
5. **ces** - Complex Exponential Smoothing. An alternative to ETS that uses complex smoothing parameters, capturing both level and "potential" of the series. Particularly useful for series with complex seasonal patterns.
6. **gum** - Generalised Univariate Model. An extension of CES that allows for more flexible lag structures. Useful when standard ETS or CES specifications are too restrictive.
7. **sma** - Simple Moving Average in state space form. Implements SMA with automatic order selection based on information criteria.
8. **ssarima** - State-Space ARIMA. Estimates ARIMA models using the state space framework, allowing for combination with other smooth package models and consistent treatment of prediction intervals.
9. **sparma** - Sparse ARMA Model in State Space form. Implements ARMA models where AR and MA orders are directly mapped to specific lags rather than expanding polynomials. Useful for modelling specific lag dependencies without the full polynomial expansion.

### Model selection functions

1. **auto.ces** - Automatic selection between seasonal and non-seasonal CES models based on information criteria.
2. **auto.ssarima** - Automatic selection between different State-Space ARIMA specifications. Tests various combinations of AR, I, and MA orders.
3. **auto.msarima** - Automatic selection between different multiple seasonal ARIMA models. Wrapper for **auto.adam()**.
4. **auto.gum** - Automatic selection of the most appropriate GUM model specification.

### Simulation functions

1. **sim.es** - Simulates data using the ETS framework with predefined or randomly generated smoothing parameters and initial values. Useful for Monte Carlo experiments and model validation.
2. **sim.ssarima** - Simulates data using the State-Space ARIMA framework with predefined or randomly generated parameters.
3. **sim.ces** - Simulates data using CES with predefined or randomly generated complex smoothing parameters.
4. **sim.gum** - Simulates data from GUM models.
5. **sim.sma** - Simulates data from Simple Moving Average models.
6. **sim.oes** - Simulates the occurrence part of ETS models. Generates probabilities of demand occurrence using various occurrence mechanisms (odds-ratio, inverse-odds-ratio, direct, general). Uses **sem.es()**

### Occurrence (intermittent demand) models

1. **oes** - Occurrence state-space Exponential Smoothing model for intermittent demand forecasting. Models the probability of non-zero demand using ETS-style state space models. Supports multiple occurrence types: fixed probability, odds ratio, inverse odds ratio, direct, and general. Can automatically select the best occurrence model.
2. **oesg** - Occurrence ETS General model. Implements the most flexible oETS_G specification where the probability is modelled as p_t = a_t/(a_t + b_t), with both a_t and b_t following separate ETS models. Provides the most flexibility for complex intermittent demand patterns.

### Scale model (GARCH style)
1. **sm** - Scale Model. Creates a model for the scale (variance) of the error term based on the provided ADAM model. Allows modelling heteroscedasticity using ETS or ARIMA structures for the scale parameter (GARCH or GAMLSS style). Only works with models estimated via maximum likelihood.

## Tools

### Utility functions

1. **msdecompose** - Multiple seasonal decomposition based on centred moving averages. Decomposes a time series into trend, multiple seasonal components, and remainder. Supports different ways how to smooth time series via the `smoother` parameter.
2. **sowhat** - Returns the ultimate answer to any question.
3. **cma** - Centred Moving Average. Used for smoothing time series and extracting trend-cycle components. This is a decomposition tool, not a forecasting function.
4. **smoothCombine** - Combines forecasts from es(), ces(), gum(), ssarima(), and sma() functions using information criteria weights. Provides a simple way to create ensemble forecasts. Left here for legacy purposes. But we don't recommend using it.

## Methods:

### Refitting and reforecasting

1. **reapply** - Reapplies the model with randomly generated parameters based on the covariance matrix of the estimated parameters. Useful for understanding parameter uncertainty and its impact on fitted values. Returns fitted paths for each parameter set.
2. **reforecast** - Produces forecasts using randomly generated parameters (via reapply) and simulated future errors. Provides a way to construct prediction intervals that account for both parameter uncertainty and future uncertainty.

### Visualisation and output

1. **plot** - Produces diagnostic plots for the model. Multiple plot types available (see documentation for plot.adam()): fitted vs actuals, standardised residuals, ACF/PACF of residuals, and more.
2. **print** - Prints a basic output of the model to the console.
3. **summary** - Provides a detailed summary of the model including parameter estimates, standard errors, and confidence intervals.
4. **xtable** - Creates LaTeX table output for the model summary. Useful for including results in academic papers.

### Coefficients and parameters

1. **coef** (coefficients) - Extracts the estimated parameters from the model.
2. **confint** - Confidence intervals for the estimated parameters.
3. **vcov** - Variance-covariance matrix of the estimated parameters. Useful for understanding parameter uncertainty.
4. **coefbootstrap** - Bootstrap estimates of the model coefficients. Provides bootstrap distributions for parameter inference.

### Fitted values and forecasts

1. **fitted** - Extracts fitted values from the model.
2. **forecast** - Produces point forecasts and prediction intervals for h steps ahead. Supports various interval types (parametric, semiparametric, nonparametric, simulated).
3. **predict** - Similar to forecast but can also produce in-sample confidence intervals for the conditional mean.
4. **actuals** - Extracts the actual (observed) values used in model estimation.

### Simulation

1. **simulate** - Simulates new data from the fitted model. Useful for scenario analysis and generating synthetic datasets.

### Residuals and forecast errors

1. **residuals** - Extracts residuals from the model. For additive error models returns e_t, for multiplicative returns log(1+e_t).
2. **rstandard** - Standardised residuals (residuals divided by their estimated standard deviation).
3. **rstudent** - Studentised residuals (leave-one-out standardised residuals).
4. **rmultistep** - Extracts 1 to h steps ahead forecast errors from the model. Returns a matrix with observations in rows and forecast horizons in columns. Useful for analysing forecast error patterns across different horizons.
5. **multicov** - Covariance matrix of 1 to h steps ahead forecast errors. Can be computed analytically, empirically, or via simulation.

### Likelihood and scoring

1. **pls** - Prediction Likelihood Score for evaluating the model on holdout data.
2. **accuracy** - Computes various accuracy measures (ME, RMSE, MAE, MPE, MAPE, etc.) for the model's forecasts.

### Information criteria and model selection

1. **AIC, BIC, AICc, BICc** - Information criteria for model selection. AICc and BICc are corrected versions for small samples.
2. **logLik** - Log-likelihood of the fitted model.
3. **pointLik** - Vector of point log-likelihoods for each in-sample observation. Needed for pAIC and other point criteria.
4. **pAIC, pAICc, pBIC, pBICc** - Point information criteria based on pointLik. Help identify observations that contribute most to model complexity. See `greybox` package documentation for details.


### Model information

1. **nobs** - Number of observations used in model estimation.
2. **nparam** - Number of estimated parameters in the model, broken down by category (estimated, provided, scale).
3. **sigma** - Residual standard deviation (scale parameter) of the model.
4. **extractScale** - Extracts the scale parameter from the model (for models using scale distributions).
5. **extractSigma** - Extracts sigma value from the model.
6. **errorType** - Extracts the type of error in the model: additive ("A") or multiplicative ("M").
7. **modelType** - Extracts the type of the estimated model (e.g., "AAdN" for ETS, or "CES" variants).
8. **modelName** - Returns the full descriptive name of the fitted model (e.g., "ARIMA(0,1,1)", "ETS(A,Ad,N)").
9. **orders** - Extracts the orders of ARIMA components (ar, i, ma). Mainly useful for ssarima, msarima, adam with ARIMA, and GUM.
10. **lags** - Extracts the lags of the model. Useful for ARIMA, GUM, and models with multiple seasonalities.

### Outliers and special handling

1. **outlierdummy** - Creates a matrix of dummy variables based on detected outliers in the residuals. Useful for identifying and handling anomalous observations.


## Installation

The stable version of the package is available on CRAN, so you can install it by running:
> install.packages("smooth")

A recent, development version, is available via github and can be installed using "remotes" in R. First, make sure that you have remotes:
> if (!require("remotes")){install.packages("remotes")}

and after that run:
> remotes::install_github("config-i1/smooth")

## Notes

The package depends on Rcpp and RcppArmadillo, which will be installed automatically.

However Mac OS users may need to install gfortran libraries in order to use Rcpp. Follow the link for the instructions: http://www.thecoatlessprofessor.com/programming/rcpp-rcpparmadillo-and-os-x-mavericks-lgfortran-and-lquadmath-error/

Sometimes after upgrade of smooth from previous versions some functions stop working. This is because C++ functions are occasionally stored in deeper unknown corners of R's mind. Restarting R usually solves the problem. If it  doesn't, completely remove smooth (uninstal + delete the folder "smooth" from R packages folder), restart R and reinstall smooth.
