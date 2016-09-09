context("Tests for ssarima() function");

# Basic SSARIMA selection
testModel <- auto.ssarima(Mcomp::M3$N1234$x, silent=TRUE);
test_that("Test if Auto SSARIMA selected correct model for N1234$x", {
    expect_equal(testModel$model, "ARIMA(0,1,0) with drift");
})

# Reuse previous SSARIMA
test_that("Reuse previous GES on N1234$x", {
    expect_equal(ssarima(Mcomp::M3$N1234$x, model=testModel, silent=TRUE)$cf, testModel$cf);
})

# Test some crazy order of SSARIMA
testModel <- ssarima(Mcomp::M3$N2568$x, ar.orders=c(1,1,0), i.orders=c(1,0,1), ma.orders=c(0,1,1), lags=c(1,6,12), h=18, holdout=TRUE, initial="o", silent=TRUE, intervals=TRUE)
test_that("Test if crazy order SSARIMA was estimated on N1234$x", {
    expect_equal(testModel$model, "SARIMA(1,1,0)[1](1,0,1)[6](0,1,1)[12]");
})

testModel <- auto.ssarima(Mcomp::M3$N2568$x, silent=TRUE);
# Test how different passed values are accepted by SSARIMA
test_that("Test initials, AR, MA and constant of SSARIMA on N2568$x", {
    expect_equal(ssarima(Mcomp::M3$N2568$x, ar.orders=c(2,0), i.orders=c(0,1), ma.orders=c(3,3), lags=c(1,12), constant=TRUE, initial=testModel$initial, silent=TRUE)$initial, testModel$initial);
    expect_equal(ssarima(Mcomp::M3$N2568$x, ar.orders=c(2,0), i.orders=c(0,1), ma.orders=c(3,3), lags=c(1,12), constant=TRUE, AR=testModel$AR, silent=TRUE)$AR, testModel$AR);
    expect_equal(ssarima(Mcomp::M3$N2568$x, ar.orders=c(2,0), i.orders=c(0,1), ma.orders=c(3,3), lags=c(1,12), constant=TRUE, transition=testModel$MA, silent=TRUE)$MA, testModel$MA);
    expect_equal(ssarima(Mcomp::M3$N2568$x, ar.orders=c(2,0), i.orders=c(0,1), ma.orders=c(3,3), lags=c(1,12), constant=testModel$constant, silent=TRUE)$constant, testModel$constant);
})

# Test exogenous (normal + updateX) with SSARIMA
x <- cbind(c(rep(0,25),1,rep(0,43)),c(rep(0,10),1,rep(0,58)));
y <- ts(c(Mcomp::M3$N1457$x,Mcomp::M3$N1457$xx),frequency=12);
testModel <- ssarima(y, h=18, holdout=TRUE, xreg=x, updateX=TRUE, silent=TRUE, cfType="aMSTFE", intervals=TRUE)
test_that("Check exogenous variables for SSARIMA on N1457", {
    expect_equal(suppressWarnings(round(ssarima(y, h=18, holdout=TRUE, xreg=x, silent=TRUE)$forecast[1],3)), 5986.654);
    expect_equal(suppressWarnings(round(forecast(testModel, h=18, holdout=FALSE)$forecast[18],3)), 2855.424);
})
