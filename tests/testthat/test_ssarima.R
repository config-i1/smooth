context("Tests for ssarima() function")

# Basic SSARIMA selection
testModel <- auto.ssarima(BJsales, silent=TRUE)
test_that("Test if Auto SSARIMA selected correct model for BJsales", {
    expect_equal(testModel$model, ssarima(BJsales, model=testModel)$model)
})

# Reuse previous SSARIMA
test_that("Reuse previous SSARIMA on BJsales", {
    expect_equal(ssarima(BJsales, model=testModel, silent=TRUE)$cf, testModel$cf)
})

# Test some crazy order of SSARIMA
test_that("Test if crazy order SSARIMA was estimated on AirPassengers", {
    skip_on_cran()
    testModel <- ssarima(AirPassengers, orders=list(ar=c(1,1,0), i=c(1,0,1),ma=c(0,1,1)),
                         lags=c(1,6,12), h=18, holdout=TRUE, initial="o", silent=TRUE, interval=TRUE)
    expect_equal(testModel$model, "SSARIMA(1,1,0)[1](1,0,1)[6](0,1,1)[12]")
})

# Test selection of exogenous with Auto.SSARIMA
test_that("Use exogenous variables for auto SSARIMAX on BJsales with selection", {
    skip_on_cran()
    testModel <- auto.ssarima(BJsales, orders=list(ar=3,i=2,ma=3), lags=1, h=18, holdout=TRUE,
                              regressors="use", silent=TRUE, xreg=xregExpander(BJsales.lead))
    expect_equal(length(testModel$initial$xreg),3)
})
