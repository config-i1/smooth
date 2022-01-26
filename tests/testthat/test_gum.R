context("Tests for gum() function")

# Basic GUM selection
testModel <- gum(BJsales, orders=c(2,1),lags=c(1,4), silent=TRUE)
test_that("Test if GUM worked on BJsales", {
    expect_equal(testModel$model, "GUM(2[1],1[4])")
})

# Reuse previous GUM
test_that("Reuse previous GUM on BJsales", {
    expect_equal(gum(BJsales, model=testModel, silent=TRUE)$cf, testModel$cf)
})

# Test some crazy order of GUM
test_that("Test if crazy order GUM was estimated on BJsales", {
    skip_on_cran()
    testModel <- gum(BJsales, orders=c(1,1,1), lags=c(1,3,5), h=18, holdout=TRUE, initial="o", silent=TRUE, interval=TRUE)
    expect_equal(testModel$model, "GUM(1[1],1[3],1[5])")
})

# Test how different passed values are accepted by GUM
test_that("Test initials, measurement, transition and persistence of GUM on AirPassengers", {
    skip_on_cran()
    testModel <- gum(AirPassengers, orders=c(1,1,1), lags=c(1,3,5), h=18, holdout=TRUE, initial="o", silent=TRUE, interval=TRUE)
    expect_equal(gum(AirPassengers, orders=c(1,1,1), lags=c(1,3,5), initial=testModel$initial, silent=TRUE)$initial, testModel$initial)
    expect_equal(gum(AirPassengers, orders=c(1,1,1), lags=c(1,3,5), measurement=testModel$measurement, silent=TRUE)$measurement, testModel$measurement)
    expect_equal(gum(AirPassengers, orders=c(1,1,1), lags=c(1,3,5), transition=testModel$transition, silent=TRUE)$transition, testModel$transition)
    expect_equal(gum(AirPassengers, orders=c(1,1,1), lags=c(1,3,5), persistence=testModel$persistence, silent=TRUE)$persistence, testModel$persistence)
})

# Test selection of exogenous with GUM
test_that("Select exogenous variables for GUMX on BJsales", {
    skip_on_cran()
    x <- BJsales.lead
    y <- BJsales
    testModel <- gum(y, h=18, holdout=TRUE, xreg=xregExpander(x), silent=TRUE, xregDo="select")
    expect_match(errorType(testModel),"A")
})

# Use automatic GUM
test_that("Use automatic GUM on BJsales", {
    skip_on_cran()
    expect_equal(auto.gum(BJsales, silent=TRUE, bounds="r")$model, "GUM(2[1])")
})
