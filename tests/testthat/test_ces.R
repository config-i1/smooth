context("Tests for ces() function");

# Basic CES selection
testModel <- auto.ces(AirPassengers, silent=TRUE);
test_that("Test CES selection on BJsales", {
    expect_match(testModel$seasonality, "partial");
})

# Reuse previous CES
test_that("Test on AirPassengers, predefined CES", {
    expect_equal(ces(AirPassengers, model=testModel, silent=TRUE)$loss, testModel$loss);
})

# Test trace cost function for CES
testModel <- ces(AirPassengers, seasonality="f", h=18, holdout=TRUE, silent=TRUE)
test_that("Test AICc of CES based on MSTFE on AirPassengers", {
    expect_equal(as.numeric(logLik(testModel)), as.numeric(testModel$logLik));
})

# Test how different passed values are accepted by CES
test_that("Test provided a and b of CES on AirPassengers", {
    expect_equal(ces(AirPassengers, seasonality="f", a=testModel$parameters$a, silent=TRUE)$parameters$a,
                 testModel$parameters$a);
    expect_equal(ces(AirPassengers, seasonality="f", b=testModel$parameters$b, silent=TRUE)$parameters$b,
                 testModel$parameters$b);
})

# Test selection of exogenous with CES
test_that("Use exogenous variables for CESX on BJsales", {
    skip_on_cran()
    testModel <- ces(BJsales, h=18, holdout=TRUE, silent=TRUE, regressors="use", xreg=BJsales.lead)
    expect_equal(length(testModel$initial$xreg),1);
})
