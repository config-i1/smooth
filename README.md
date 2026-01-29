# smooth

R:
[![CRAN_Status_Badge](https://www.r-pkg.org/badges/version/smooth)](https://cran.r-project.org/package=smooth)
[![Downloads](https://cranlogs.r-pkg.org/badges/smooth)](https://cran.r-project.org/package=smooth)
[![R-CMD-check](https://github.com/config-i1/smooth/actions/workflows/test.yml/badge.svg)](https://github.com/config-i1/smooth/actions/workflows/test.yml)

Python:
[![PyPI - Downloads](https://img.shields.io/pypi/dm/smooth.svg)](https://pypi.org/project/smooth/)

The package _smooth_ contains several smoothing (exponential and not) functions that are used in forecasting.

![hex-sticker of the smooth package for R](https://github.com/config-i1/smooth/blob/master/man/figures/smooth-web.png?raw=true)

Here is the list of the included functions:

### Main forecasting functions

1. **adam** - Advanced Dynamic Adaptive Model, the main unified framework implementing ETS (Error-Trend-Seasonal), ARIMA (AutoRegressive Integrated Moving Average), and regression with exogenous variables. Supports multiple seasonalities, various error distributions (Normal, Laplace, S, Generalised Normal, Log-Normal, Inverse Gaussian, Gamma), and different loss functions. This is the recommended function for most forecasting tasks.
2. **auto.adam** - Automatic model selection for ADAM. Selects the best distribution, ARIMA orders (if requested), and can detect and handle outliers. Supports parallel computation for faster estimation when testing multiple distributions.
3. **es** - Exponential Smoothing (ETS) function. Implements all 30 ETS model types with automatic model selection using branch and bound algorithm. Supports exogenous variables (ETX), holdout validation, multiple loss functions including trace forecast based ones, and AIC-weighted forecast combinations.
4. **ces** - Complex Exponential Smoothing. An alternative to ETS that uses complex smoothing parameters, capturing both level and "potential" of the series. Particularly useful for series with complex seasonal patterns.
5. **gum** - Generalised Univariate Model. An extension of CES that allows for more flexible lag structures. Useful when standard ETS or CES specifications are too restrictive.
6. **sma** - Simple Moving Average in state space form. Implements SMA with automatic order selection based on information criteria.
7. **ssarima** - State-Space ARIMA. Estimates ARIMA models using the state space framework, allowing for combination with other smooth package models and consistent treatment of prediction intervals.
8. **msarima** - Multiple Seasonal ARIMA. Extends ssarima to handle multiple seasonal patterns (e.g., hourly data with daily, weekly, and annual seasonality). More computationally efficient than ssarima for complex seasonal structures.
9. **sparma** - Sparse ARMA Model in State Space form. Implements ARMA models where AR and MA orders are directly mapped to specific lags rather than expanding polynomials. Useful for modelling specific lag dependencies without the full polynomial expansion.

### Model selection functions

10. **auto.ces** - Automatic selection between seasonal and non-seasonal CES models based on information criteria.
11. **auto.ssarima** - Automatic selection between different State-Space ARIMA specifications. Tests various combinations of AR, I, and MA orders.
12. **auto.msarima** - Automatic selection between different multiple seasonal ARIMA models.
13. **auto.gum** - Automatic selection of the most appropriate GUM model specification.

### Simulation functions

14. **sim.es** - Simulates data using the ETS framework with predefined or randomly generated smoothing parameters and initial values. Useful for Monte Carlo experiments and model validation.
15. **sim.ssarima** - Simulates data using the State-Space ARIMA framework with predefined or randomly generated parameters.
16. **sim.ces** - Simulates data using CES with predefined or randomly generated complex smoothing parameters.
17. **sim.gum** - Simulates data from GUM models.
18. **sim.sma** - Simulates data from Simple Moving Average models.
19. **sim.oes** - Simulates the occurrence part of ETS models. Generates probabilities of demand occurrence using various occurrence mechanisms (odds-ratio, inverse-odds-ratio, direct, general).

### Occurrence (intermittent demand) models

20. **oes** - Occurrence state-space Exponential Smoothing model for intermittent demand forecasting. Models the probability of non-zero demand using ETS-style state space models. Supports multiple occurrence types: fixed probability, odds ratio, inverse odds ratio, direct, and general. Can automatically select the best occurrence model.
21. **oesg** - Occurrence ETS General model. Implements the most flexible iETS_G specification where the probability is modelled as p_t = a_t/(a_t + b_t), with both a_t and b_t following separate ETS models. Provides the most flexibility for complex intermittent demand patterns.

### Refitting and reforecasting

22. **reapply** - Reapplies the model with randomly generated parameters based on the covariance matrix of the estimated parameters. Useful for understanding parameter uncertainty and its impact on fitted values. Returns fitted paths for each parameter set.
23. **reforecast** - Produces forecasts using randomly generated parameters (via reapply) and simulated future errors. Provides a way to construct prediction intervals that account for both parameter uncertainty and future uncertainty.

### Utility functions

24. **smoothCombine** - Combines forecasts from es(), ces(), gum(), ssarima(), and sma() functions using information criteria weights. Provides a simple way to create ensemble forecasts.
25. **rmultistep** - Extracts 1 to h steps ahead forecast errors from the model. Returns a matrix with observations in rows and forecast horizons in columns. Useful for analysing forecast error patterns across different horizons.
26. **cma** - Centred Moving Average. Used for smoothing time series and extracting trend-cycle components. This is a decomposition tool, not a forecasting function.
27. **msdecompose** - Multiple seasonal decomposition based on centred moving averages. Decomposes a time series into trend, multiple seasonal components, and remainder.
28. **sowhat** - Returns the ultimate answer to any question.

Available methods:

### Information criteria and model selection

1. **AIC, BIC, AICc, BICc** - Information criteria for model selection. AICc and BICc are corrected versions for small samples.

### Coefficients and parameters

2. **coef** (coefficients) - Extracts the estimated parameters from the model.
3. **confint** - Confidence intervals for the estimated parameters.
4. **vcov** - Variance-covariance matrix of the estimated parameters. Useful for understanding parameter uncertainty.
5. **coefbootstrap** - Bootstrap estimates of the model coefficients. Provides bootstrap distributions for parameter inference.

### Model diagnostics and extraction

6. **multicov** - Covariance matrix of 1 to h steps ahead forecast errors. Can be computed analytically, empirically, or via simulation.
7. **errorType** - Extracts the type of error in the model: additive ("A") or multiplicative ("M").
8. **modelType** - Extracts the type of the estimated model (e.g., "AAdN" for ETS, or "CES" variants).
9. **modelName** - Returns the full descriptive name of the fitted model (e.g., "ARIMA(0,1,1)", "ETS(A,Ad,N)").
10. **orders** - Extracts the orders of ARIMA components (ar, i, ma). Mainly useful for ssarima, msarima, adam with ARIMA, and GUM.
11. **lags** - Extracts the lags of the model. Useful for ARIMA, GUM, and models with multiple seasonalities.

### Fitted values and forecasts

12. **fitted** - Extracts fitted values from the model.
13. **forecast** - Produces point forecasts and prediction intervals for h steps ahead. Supports various interval types (parametric, semiparametric, nonparametric, simulated).
14. **predict** - Similar to forecast but can also produce in-sample confidence intervals for the conditional mean.
15. **actuals** - Extracts the actual (observed) values used in model estimation.

### Residuals

16. **residuals** - Extracts residuals from the model. For additive error models returns e_t, for multiplicative returns log(1+e_t).
17. **rstandard** - Standardised residuals (residuals divided by their estimated standard deviation).
18. **rstudent** - Studentised residuals (leave-one-out standardised residuals).

### Likelihood and scoring

19. **logLik** - Log-likelihood of the fitted model.
20. **pointLik** - Vector of individual log-likelihoods for each in-sample observation. Useful for identifying problematic observations.
21. **pAIC** - Point AIC values based on pointLik. Helps identify observations that contribute most to model complexity.
22. **pls** - Prediction Likelihood Score for evaluating the model on holdout data.
23. **accuracy** - Computes various accuracy measures (ME, RMSE, MAE, MPE, MAPE, etc.) for the model's forecasts.

### Model information

24. **nobs** - Number of observations used in model estimation.
25. **nparam** - Number of estimated parameters in the model, broken down by category (estimated, provided, scale).
26. **sigma** - Residual standard deviation (scale parameter) of the model.
27. **extractScale** - Extracts the scale parameter from the model (for models using scale distributions).
28. **extractSigma** - Extracts sigma value from the model.

### Outliers and special handling

29. **outlierdummy** - Creates a matrix of dummy variables based on detected outliers in the residuals. Useful for identifying and handling anomalous observations.

### Visualisation and output

30. **plot** - Produces diagnostic plots for the model. Multiple plot types available (see documentation for plot.adam() and plot.smooth()): fitted vs actuals, standardised residuals, ACF/PACF of residuals, and more.
31. **print** - Prints a summary of the model to the console.
32. **summary** - Provides a detailed summary of the model including parameter estimates, standard errors, and significance tests.
33. **xtable** - Creates LaTeX table output for the model summary. Useful for including results in academic papers.

### Simulation

34. **simulate** - Simulates new data from the fitted model. Useful for scenario analysis and generating synthetic datasets.

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
