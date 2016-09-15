# smooth
The package contains several smoothing (exponential and not) functions that are used in forecasting

Here is the list of functions:

1. es - the ETS function. It can handle exogenous variables and has a handy "holdout" parameter. There are several cost function implemented, including trace forecast based ones. Model selection is done via branch and bound algorithm and there's a possibility to use AIC weights in order to produce combined forecasts. Finally, all the possible ETS functions are implemented here.
2. ces - Complex Exponential Smoothing. Function estimates CES and makes forecast. See documentation for details.
3. ges - Generalised Exponential Smoothing. Next step from CES. The paper on this is in the process.
4. ssarima - SARIMA estimated in state-space framework. Allows multiple seasonalities.
5. auto.ces - selection between seasonal and non-seasonal CES models.
6. auto.ssarima - selection between different ARIMA models.
7. sim.ets - simulation of data using ETS framework with a predefined (or random) smoothing parameters and initial values.
8. iss - Intermittent data state-space model. This function models the part with data occurrences using one of three methods.
9. sma - Simple Moving Average in state-space form.

Future works:

10. sim.ces, sim.ges, sim.ssarima - simulation functions for CES, GES and SSARIMA respectively.
11. nus - Non-uniform Smoothing. The estimation method used in order to update parameters of a regression model.
12. sofa.ts - Survival of the fittest algorithm applied to state-space models.

## Installation

For a quick and easy installation of the package firstly install "devtools" in R:
> if (!require("devtools")){install.packages("devtools")}

And after that run:

> devtools::install_github("config-i1/smooth")

## Notes

The package depends on Rcpp and RcppArmadillo, which will be installed automatically.

However Mac OS users may need to install gfortran libraries in order to use Rcpp. Follow the link for the instructions: http://www.thecoatlessprofessor.com/programming/rcpp-rcpparmadillo-and-os-x-mavericks-lgfortran-and-lquadmath-error/

Sometimes after upgrade of smooth from previous versions some functions stop working. This is because C++ functions are occasionally stored in depper unknown coners of R's mind. Restarting R usually solves the problem.
