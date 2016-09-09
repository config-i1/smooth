context("Tests for ces() function");

# Basic CES selection
testModel <- auto.ces(Mcomp::M3$N1234$x, silent=TRUE);
test_that("Test CES selection on N1234$x", {
    expect_match(testModel$model, "n");
})

# Reuse previous CES
test_that("Test on N1234$x, predefined CES", {
    expect_equal(ces(Mcomp::M3$N1234$x, model=testModel, silent=TRUE)$cf, testModel$cf);
})

# Test trace cost function for CES
testModel <- ces(Mcomp::M3$N2568$x, seasonality="f", h=18, holdout=TRUE, cfType="MSTFE", silent=TRUE, intervals=TRUE)
test_that("Test AICc of CES based on MSTFE on N2568$x", {
    expect_equal(AICc(testModel), testModel$ICs["AICc"]);
})

# Test how different passed values are accepted by CES
test_that("Test initials, A and B of CES on N2568$x", {
    expect_equal(ces(Mcomp::M3$N2568$x, seasonality="f", initial=testModel$initial, silent=TRUE)$initial, testModel$initial);
    expect_equal(ces(Mcomp::M3$N2568$x, seasonality="f", A=testModel$A, silent=TRUE)$A, testModel$A);
    expect_equal(ces(Mcomp::M3$N2568$x, seasonality="f", B=testModel$B, silent=TRUE)$B, testModel$B);
})

# Test exogenous (normal + updateX) with CES
x <- cbind(c(rep(0,25),1,rep(0,43)),c(rep(0,10),1,rep(0,58)));
y <- ts(c(Mcomp::M3$N1457$x,Mcomp::M3$N1457$xx),frequency=12);
testModel <- ces(y, h=18, holdout=TRUE, xreg=x, updateX=TRUE, silent=TRUE, cfType="aMSTFE", intervals=TRUE, intervalsType="s")
test_that("Check exogenous variables for CES on N1457", {
    expect_equal(suppressWarnings(round(ces(y, h=18, holdout=TRUE, xreg=x, silent=TRUE)$forecast[1],3)), 5611.054);
    expect_equal(suppressWarnings(round(forecast(testModel, h=18, holdout=FALSE)$forecast[18],3)), 6181.582);
})
