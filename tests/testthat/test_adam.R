context("Tests for ADAM");

#### Basic ETS stuff ####
# Basic ADAM selection
testModel <- adam(Mcomp::M3[[1234]], "ZZZ");
test_that("ADAM ETS(ZZZ) selection on N1234", {
    expect_match(modelType(testModel), "MMN");
})

# Basic ADAM selection on 2568
testModel <- adam(Mcomp::M3[[2568]], "ZZZ");
test_that("ADAM ETS(ZZZ) selection on N2568", {
    expect_match(modelType(testModel), "MAM");
})

# Full ADAM selection
testModel <- adam(Mcomp::M3[[1234]], "FFF");
test_that("ADAM ETS(FFF) selection on N1234", {
    expect_match(modelType(testModel), "MMN");
})

# ADAM with specified pool
testModel <- adam(Mcomp::M3[[1234]], c("AAA","ANN","MAN","MAM"));
test_that("ADAM selection with a pool on N1234", {
    expect_match(modelType(testModel), "MAN");
})

# ADAM forecasts with simulated interval
testForecast <- forecast(testModel,h=8,interval="sim",level=c(0.9,0.95));
test_that("ADAM forecast with simulated interval", {
    expect_equal(ncol(testForecast$lower), 2);
})

# ADAM combination
testModel <- adam(Mcomp::M3[[1234]], "CCC");
test_that("ADAM ETS(CCC) on N1234", {
    expect_match(modelType(testModel), "CCC");
})

# ADAM forecasts with approximated interval
testForecast <- forecast(testModel,h=8,interval="app",level=c(0.9,0.95),side="upper");
test_that("ADAM forecast with simulated interval", {
    expect_equal(ncol(testForecast$lower), 2);
})


#### Advanced losses for ADAM ####
# ADAM with dalaplace
testModel <- adam(Mcomp::M3[[1234]], "MAN", distribution="dalaplace", alpha=0.05);
test_that("ADAM ETS(MAN) with asymmetric Laplace on N1234", {
    expect_match(testModel$distribution, "dalaplace");
})

# ADAM with GN distribution
testModel <- adam(Mcomp::M3[[1234]], "MAN", distribution="dgnorm");
test_that("ADAM ETS(MAN) with Generalised Normal on N1234", {
    expect_match(testModel$distribution, "dgnorm");
})

# ADAM with MSE
testModel <- adam(Mcomp::M3[[1234]], "MAN", loss="MSE");
test_that("ADAM ETS(MAN) with MSE on N1234", {
    expect_match(testModel$loss, "MSE");
})

# ADAM with MSEh
testModel <- adam(Mcomp::M3[[1234]], "MAN", loss="MSEh");
test_that("ADAM ETS(MAN) with MSE on N1234", {
    expect_match(testModel$loss, "MSEh");
})

# ADAM with GTMSE
testModel <- adam(Mcomp::M3[[1234]], "MAN", loss="GTMSE");
test_that("ADAM ETS(MAN) with GTMSE on N1234", {
    expect_match(testModel$loss, "GTMSE");
})

# ADAM with GPL
testModel <- adam(Mcomp::M3[[1234]], "MAN", loss="GPL");
test_that("ADAM ETS(MAN) with GPL on N1234", {
    expect_match(testModel$loss, "GPL");
})

# ADAM with LASSO
testModel <- adam(Mcomp::M3[[1234]], "MAN", loss="LASSO", lambda=0.5);
test_that("ADAM ETS(MAN) with LASSO on N1234", {
    expect_match(testModel$loss, "LASSO");
})

# ADAM with custom loss function
loss <- function(actual, fitted, B){
    return(sum(abs(actual-fitted)^3));
}
testModel <- adam(Mcomp::M3[[1234]], "AAN", loss=loss);
test_that("ADAM ETS(AAN) with custom loss on N1234", {
    expect_match(testModel$loss, "custom");
})


#### ETS + occurrence model ####
# Generate intermittent data
x <- sim.oes("MNN", 120, frequency=12, occurrence="general", persistence=0.01, initial=2, initialB=1)
x <- sim.es("MNN", 120, frequency=12, probability=x$probability, persistence=0.1)

# iETS(M,N,N)_G
testModel <- adam(x$data, "MNN", occurrence="general")
test_that("ADAM iETS(MNN) with general occurrence", {
    expect_match(testModel$occurrence$occurrence, "general");
})

# iETS(M,M,M)_A
testModel <- adam(x$data, "MMM", occurrence="direct")
test_that("ADAM iETS(MMM) with direct occurrence", {
    expect_match(errorType(testModel), "M");
})

# iETS(M,M,N)_A
testModel <- adam(x$data, "MMN", occurrence="auto")
test_that("ADAM iETS(MMN) with auto occurrence", {
    expect_match(errorType(testModel), "M");
})

# iETS(Z,Z,N)_A
testModel <- adam(x$data, "ZZN", occurrence="auto")
test_that("ADAM iETS(MMN) with auto occurrence", {
    expect_true(is.occurrence(testModel$occurrence));
})

# Forecasts from the model
testForecast <- forecast(testModel, h=18, interval="semi")
test_that("Froecast from ADAM iETS(ZZZ)", {
    expect_true(is.adam(testForecast$model));
})


#### ETS with several seasonalities ####
# Double seasonality on N2568
testModel <- adam(Mcomp::M3[[2568]]$x, "YYY", lags=c(1,3,12), h=18);
test_that("ADAM ETS(YYY) with double seasonality on N2568", {
    expect_identical(testModel$lags, c(1,3,12));
})

# Double seasonality on N2568
testModel <- adam(Mcomp::M3[[2568]]$x, "FFF", lags=c(1,3,12), h=18, initial="backcasting");
test_that("ADAM ETS(FFF) + backcasting with double seasonality on N2568", {
    expect_identical(testModel$lags, c(1,3,12));
})

# Double seasonality on N2568
testModel <- adam(Mcomp::M3[[2568]]$x, "CCC", lags=c(1,3,12), h=18);
test_that("ADAM ETS(CCC) with double seasonality on N2568", {
    expect_identical(testModel$models[[1]]$lags, c(1,3,12));
})


#### ETSX / Regression + formula ####
# ETSX on N2568
xreg <- data.frame(y=Mcomp::M3[[2568]]$x, x=factor(temporaldummy(Mcomp::M3[[2568]]$x)[,-1] %*% c(1:11)))
testModel <- adam(xreg, "MMN", h=18, holdout=TRUE);
test_that("ADAM ETSX(MMN) on N2568", {
    expect_false(ncol(testModel$data)==1);
})

# ETSX selection on N2568
testModel <- adam(xreg, "ZZZ", h=18, holdout=TRUE, regressors="select");
test_that("ADAM ETSX(ZZZ) + xreg selection on N2568", {
    expect_equal(testModel$regressors,"use");
})

# ETSX adaption on N2568
testModel <- adam(xreg, "MMN", h=18, holdout=TRUE, regressors="adapt");
test_that("ADAM ETSX(MMN) + xreg adapt on N2568", {
    expect_match(testModel$regressors, "adapt");
})

# Forecast from ETSX with formula
testForecast <- forecast(testModel, h=18, newxreg=tail(xreg, 18), interval="simulated");
test_that("Forecast for ADAM adaptive regression on N2568", {
    expect_equal(testForecast$level, 0.95);
})

# ETSX with formula
testModel <- adam(xreg, "MMN", h=18, holdout=TRUE, formula=y~x, distribution="dnorm");
test_that("ADAM ETSX(MMN) + xreg formula on N2568", {
    expect_match(testModel$regressors, "use");
})

# Forecast from ETSX with formula
testForecast <- forecast(testModel, h=18, newxreg=tail(xreg, 18), interval="nonp");
test_that("Forecast for ADAM ETSX(MMN) + xreg formula on N2568", {
    expect_equal(testForecast$level, 0.95);
})

# Pure regression
testModel <- adam(xreg, "NNN", h=18, holdout=TRUE, formula=y~x);
test_that("ADAM regression (ALM) on N2568", {
    expect_equal(modelType(testModel),"NNN");
})


#### ETS + ARIMA / ARIMA + ARIMAX ####
### ETS + ARIMA
# ETS(ANN) + ARIMA(0,2,2)
testModel <- adam(Mcomp::M3[[1234]], "ANN", orders=c(0,2,2));
test_that("ADAM ETS(ANN) + ARIMA(0,2,2) on N1234", {
    expect_match(modelType(testModel), "ANN");
})

# ETS(ANN) + ARIMA(0,2,2) backcasting
testModel <- adam(Mcomp::M3[[1234]], "ANN", orders=c(1,1,2), initial="backcasting");
test_that("ADAM ETS(ANN) + ARIMA(0,2,2) with backcasting on N1234", {
    expect_match(modelType(testModel), "ANN");
})

# ETS(ZZZ) + ARIMA(0,2,2)
testModel <- adam(Mcomp::M3[[1234]], "ZZZ", orders=c(2,0,2), distribution="dlnorm");
test_that("ADAM ETS(ZZZ) + ARIMA(0,2,2) with logN on N1234", {
    expect_match(testModel$distribution, "dlnorm");
})

# ETS(ZZZ) + SARIMA(2,1,2)(2,1,1)[12]
testModel <- adam(Mcomp::M3[[2568]], "ZZZ", orders=list(ar=c(2,2),i=c(1,1), ma=c(2,1)), distribution="ds");
test_that("ADAM ETS(ZZZ) + SARIMA(2,1,2)(2,1,1)[12] with logS on N2568", {
    expect_match(testModel$distribution, "ds");
})

# Forecast from ETS(ZZZ) + SARIMA(2,1,2)(2,1,1)[12]
testForecast <- forecast(testModel, h=18, interval="prediction", side="upper");
test_that("Forecast of ADAM ETS(ZZZ) + SARIMA(2,1,2)(2,1,1)[12] with logS", {
    expect_match(testForecast$side, "upper");
})

### ARIMA / ARIMAX
# Pure SARIMA(2,1,2)(2,1,1)[12], Normal
testModel <- adam(Mcomp::M3[[2568]], "NNN", orders=list(ar=c(2,2),i=c(1,1), ma=c(2,2)), distribution="dgnorm");
test_that("ADAM SARIMA(2,1,2)(2,1,2)[12] with Logistic on N2568", {
    expect_match(testModel$distribution, "dgnorm");
})

# Forecast from SARIMA(2,1,2)(2,1,2)[12]
testForecast <- forecast(testModel, h=18, interval="approximate", side="lower");
test_that("Forecast of ADAM SARIMA(2,1,2)(2,1,2)[12]", {
    expect_match(testForecast$side, "lower");
})

# ARIMAX
testModel <- adam(xreg, "NNN", h=18, orders=list(ar=c(2,0),i=c(1,0), ma=c(2,1)), holdout=TRUE, formula=y~x);
test_that("ADAM SARIMAX on N2568", {
    expect_match(testModel$distribution, "dnorm");
})

# ARIMAX with dynamic xreg
testModel <- adam(xreg, "NNN", h=18, orders=list(ar=c(2,0),i=c(1,0), ma=c(2,1)), holdout=TRUE, formula=y~x, regressors="adapt");
test_that("ADAM SARIMAX with dynamic xreg on N2568", {
        expect_equal(length(testModel$persistence), 15);
})

#### Provided initial / persistence / phi / arma / B / reuse the model ####
### Initials
# ETS(MMM) with provided level
testModel <- adam(Mcomp::M3[[2568]], "MMM", initial=list(level=5000));
test_that("ADAM ETS(MMM) with provided level on N2568", {
    expect_false(testModel$initialEstimated["level"]);
})

# ETS(MMM) with provided trend
testModel <- adam(Mcomp::M3[[2568]], "MMM", initial=list(trend=1));
test_that("ADAM ETS(MMM) with provided trend on N2568", {
    expect_false(testModel$initialEstimated["trend"]);
})

# ETS(MMM) with provided seasonal
testModel <- adam(Mcomp::M3[[2568]], "MMM", initial=list(seasonal=Mcomp::M3[[2568]]$x[1:12]));
test_that("ADAM ETS(MMM) with provided seasonal components on N2568", {
    expect_false(testModel$initialEstimated["seasonal"]);
})

# ETSX(MMN) with provided xreg initials
testModel <- adam(xreg, "MMN", h=18, holdout=TRUE, formula=y~x,
                  initial=list(xreg=c(-0.35,-.34,.27,-.46,.07,-0.28,-0.24,0.05,-0.28,-0.34,-0.01)));
test_that("ADAM ETSX(MMN) with provided xreg initials on N2568", {
    expect_false(testModel$initialEstimated["xreg"]);
})

# ETS(ANN) + ARIMA(0,2,2) with provided initials for ARIMA
testModel <- adam(Mcomp::M3[[1234]], "ANN", orders=c(0,2,2), initial=list(arima=Mcomp::M3[[1234]]$x[1:2]));
test_that("ADAM ETS(ANN) + ARIMA(0,2,2) with initials for ARIMA on N1234", {
    expect_false(testModel$initialEstimated["arima"]);
})

# All provided initials
testModel <- adam(xreg, "MMM", formula=y~x, orders=c(0,0,2), lags=c(1,12));
testModel <- adam(xreg, "MMM", formula=y~x, orders=c(0,0,2), lags=c(1,12), initial=testModel$initial);
test_that("ADAM ETSX(MMM)+ARIMA(0,0,2) with provided initials on N2568", {
    expect_true(all(!testModel$initialEstimated));
})

### Persistence
# ETS(MMM) with provided alpha
testModel <- adam(Mcomp::M3[[2568]], "MMM", persistence=list(alpha=0.1));
test_that("ADAM ETS(MMM) with provided alpha on N2568", {
    expect_equivalent(testModel$persistence["alpha"],0.1);
})

# ETS(MMM) with provided beta
testModel <- adam(Mcomp::M3[[2568]], "MMM", persistence=list(beta=0.1));
test_that("ADAM ETS(MMM) with provided beta on N2568", {
    expect_equivalent(testModel$persistence["beta"],0.1);
})

# ETS(MMM) with provided gamma
testModel <- adam(Mcomp::M3[[2568]], "MMM", persistence=list(gamma=0.1));
test_that("ADAM ETS(MMM) with provided gamma on N2568", {
    expect_equivalent(testModel$persistence["gamma"],0.1);
})

# ETS(MMN) with provided deltas
testModel <- adam(xreg, "MMN", formula=y~x, persistence=list(delta=0.01), regressors="adapt");
test_that("ADAM ETS(MMN) with provided deltas on N2568", {
    expect_equivalent(testModel$persistence[substr(names(testModel$persistence),1,5)=="delta"],rep(0.01,12));
})

### Phi
# ETS(MMdM) with provided phi
testModel <- adam(Mcomp::M3[[2568]], "MMdM", phi=0.99);
test_that("ADAM ETS(MMdM) with provided phi on N2568", {
    expect_equivalent(testModel$phi,0.99);
})

### arma parameters
# Provided AR parameters
testModel <- adam(Mcomp::M3[[2568]], "MMM", orders=c(2,0,2), arma=list(ar=c(0.2,0.3)));
test_that("ADAM ETS(MMM)+ARIMA(2,0,2) with provided AR on N2568", {
    expect_equivalent(testModel$arma$ar,c(0.2,0.3));
})

# Provided MA parameters
testModel <- adam(Mcomp::M3[[2568]], "MMM", orders=c(2,0,2), arma=list(ma=c(-0.2,-0.4)));
test_that("ADAM ETS(MMM)+ARIMA(2,0,2) with provided MA on N2568", {
    expect_equivalent(testModel$arma$ma,c(-0.2,-0.4));
})

# Provided ARMA parameters
testModel <- adam(Mcomp::M3[[2568]], "MMM", orders=c(2,0,2), arma=list(ar=c(0.2,0.3), ma=c(-0.2,-0.4)));
test_that("ADAM ETS(MMM)+ARIMA(2,0,2) with provided ARMA on N2568", {
    expect_equivalent(testModel$arma$ar,c(0.2,0.3));
    expect_equivalent(testModel$arma$ma,c(-0.2,-0.4));
})

### B
# Provided starting parameters
testModel <- adam(Mcomp::M3[[2568]], "NNN", orders=c(2,0,2));
testModel <- adam(Mcomp::M3[[2568]], "NNN", orders=c(2,0,2), B=testModel$B);
test_that("ADAM ARIMA(2,0,2) with provided B on N2568", {
    expect_equivalent(testModel$model,"ARIMA(2,0,2)");
})

### Model reused
# Reuse ETS
testModel <- adam(Mcomp::M3[[2568]], "MMdM");
testModelNew <- adam(Mcomp::M3[[2568]], testModel);
test_that("Reuse ADAM ETS(MMdM) on N2568", {
    expect_equal(testModel$model,testModelNew$model);
    expect_equal(nparam(testModelNew),1);
})

# Reuse ARIMA
testModel <- adam(Mcomp::M3[[2568]], "NNN", h=18, orders=list(ar=c(2,0),i=c(1,0), ma=c(2,1)), holdout=TRUE);
testModelNew <- adam(Mcomp::M3[[2568]], testModel);
test_that("Reuse ADAM SARIMA(2,1,2)(0,0,1)[12] on N2568", {
    expect_equal(testModel$model,testModelNew$model);
    expect_equal(nparam(testModelNew),1);
})

# Reuse ARIMAX
testModel <- adam(xreg, "NNN", h=18, orders=list(ar=c(2,0),i=c(1,0), ma=c(2,1)), holdout=TRUE, formula=y~x, regressors="adapt");
testModelNew <- adam(xreg, testModel);
test_that("Reuse ADAM SARIMAX(2,1,2)(0,0,1)[12] with dynamic xreg on N2568", {
    expect_equal(testModel$persistence,testModelNew$persistence);
    expect_equal(nparam(testModelNew),1);
})

# Reuse ETSX + ARIMA
testModel <- adam(xreg, "ANN", h=18, orders=list(ar=c(2,0),i=c(1,0), ma=c(2,1)), holdout=TRUE, formula=y~x);
testModelNew <- adam(xreg, testModel);
test_that("Reuse ADAM ETSX(ANN)+SARIMA(2,1,2)(0,0,1)[12] on N2568", {
    expect_equal(testModel$persistence,testModelNew$persistence);
    expect_equal(nparam(testModelNew),1);
})


#### auto.adam ####
# Select the best distribution for ETS(ZZZ) on 2568
testModel <- auto.adam(Mcomp::M3[[2568]], "ZZZ");
test_that("Best auto.adam on N2568", {
    expect_match(testModel$loss, "likelihood");
})

# Outliers detection for ETS on series N291 of M1 in parallel
testModel <- auto.adam(Mcomp::M1[[291]], "ZZZ", outliers="use");
test_that("Detect outliers for ETS(ZZZ) on N291", {
    expect_match(testModel$loss, "likelihood");
})

# Best ARIMA on the 2568
testModel <- auto.adam(Mcomp::M3[[2568]], "NNN", orders=list(ar=c(3,2),i=c(2,1),ma=c(3,2),select=TRUE));
test_that("Best auto.adam ARIMA on N2568", {
    expect_match(testModel$loss, "likelihood");
})

# Outliers detection for ARIMA on series N291 of M1 in parallel
testModel <- auto.adam(Mcomp::M1[[291]], "NNN", orders=list(ar=c(3,2),i=c(2,1),ma=c(3,2),select=TRUE),
                       outliers="use");
test_that("Detect outliers for ARIMA on N291", {
    expect_false(ncol(testModel$data)==1);
})

# Best ETS+ARIMA+Regression on the 2568
testModel <- auto.adam(xreg, "ZZZ", orders=list(ar=c(3,2),i=c(2,1),ma=c(3,2),select=TRUE),
                       regressors="select", initial="back");
test_that("Best auto.adam ETS+ARIMA+Regression on N2568", {
    expect_match(testModel$loss, "likelihood");
})

# Summary of the best model
testSummary <- summary(testModel);
test_that("Summary of the produced ADAM model", {
    expect_match(testModel$loss, "likelihood");
})

# Best ETS+ARIMA+Regression on the 2568
testModel <- auto.adam(xreg, "ZZZ", orders=list(ar=c(3,2),i=c(2,1),ma=c(3,2),select=TRUE),
                       outliers="use", regressors="use", initial="back");
test_that("Best auto.adam ETS+ARIMA+Regression+outliers on N2568", {
    expect_match(testModel$loss, "likelihood");
})
