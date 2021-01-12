context("Tests for ssarima() function")

# Basic SSARIMA selection
testModel <- auto.ssarima(Mcomp::M3$N1234$x, silent=TRUE)
test_that("Test if Auto SSARIMA selected correct model for N1234$x", {
    expect_equal(testModel$model, "ARIMA(0,1,3) with drift")
})

# Reuse previous SSARIMA
test_that("Reuse previous SSARIMA on N1234$x", {
    expect_equal(ssarima(Mcomp::M3$N1234$x, model=testModel, silent=TRUE)$cf, testModel$cf)
})

# Test some crazy order of SSARIMA
test_that("Test if crazy order SSARIMA was estimated on N1234$x", {
    skip_on_cran()
    testModel <- ssarima(Mcomp::M3$N2568$x, orders=NULL, ar.orders=c(1,1,0), i.orders=c(1,0,1), ma.orders=c(0,1,1), lags=c(1,6,12), h=18, holdout=TRUE, initial="o", silent=TRUE, interval=TRUE)
    expect_equal(testModel$model, "SARIMA(1,1,0)[1](1,0,1)[6](0,1,1)[12]")
})

# Combine SSARIMA
test_that("Test if combined ARIMA works", {
    skip_on_cran()
    testModel <- auto.ssarima(Mcomp::M3$N2568$x, combine=TRUE, silent=TRUE, ic="AIC")
    expect_match(testModel$model, "combine")
})

# Test selection of exogenous with Auto.SSARIMA
test_that("Select exogenous variables for auto SSARIMAX on N1457 with selection", {
    skip_on_cran()
    x <- cbind(c(rep(0,25),1,rep(0,43)),c(rep(0,10),1,rep(0,58)))
    y <- ts(c(Mcomp::M3$N1457$x,Mcomp::M3$N1457$xx),frequency=12)
    testModel <- auto.ssarima(y, orders=list(ar=3,i=2,ma=3), lags=1, h=18, holdout=TRUE, xreg=xregExpander(x), xregDo="select", silent=TRUE)
    expect_equal(ncol(testModel$xreg),2)
})
