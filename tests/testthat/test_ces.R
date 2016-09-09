context("Tests for ces() function");

# Basic CES selection
testModel <- auto.ces(Mcomp::M3$N1234$x, silent=TRUE);
test_that("Test CES selection on N1234$x", {
    expect_match(testModel$model, "n");
})

# Reuse previous CES
test_that("Test on N1234$x, predefined CES", {
    expect_equal(round(ces(Mcomp::M3$N1234$x, model=testModel, silent=TRUE)$cf,2), 5152.93);
})

# Test trace cost function for CES
testModel <- ces(Mcomp::M3$N2568$x, seasonality="f", h=18, holdout=TRUE, cfType="MSTFE", silent=TRUE)
test_that("Test AICc of CES based on MSTFE on N2568$x", {
    expect_equal(as.vector(round(AICc(testModel),2)), 2009.96);
})

# Test how different passed values are accepted by CES
test_that("Test initials, A and B of CES on N2568$x", {
    expect_equal(as.vector(round(ces(Mcomp::M3$N2568$x, seasonality="f", initial=testModel$initial, silent=TRUE)$initial[1],3)), 7995.421);
    expect_equal(as.vector(round(ces(Mcomp::M3$N2568$x, seasonality="f", A=testModel$A, silent=TRUE)$A,5)), 1.26474+1.00515i);
    expect_equal(as.vector(round(ces(Mcomp::M3$N2568$x, seasonality="f", B=testModel$B, silent=TRUE)$B,5)), 1.3449+0.99298i);
})

# Test exogenous (normal + updateX) with ETS
x <- cbind(c(rep(0,25),1,rep(0,43)),c(rep(0,10),1,rep(0,58)));
y <- ts(c(Mcomp::M3$N1457$x,Mcomp::M3$N1457$xx),frequency=12);
testModel <- ces(y, h=18, holdout=TRUE, xreg=x, updateX=TRUE, silent=TRUE, cfType="aMSTFE")
test_that("Check exogenous variables for ETS on N1457", {
    expect_equal(suppressWarnings(round(ces(y, h=18, holdout=TRUE, xreg=x, silent=TRUE)$forecast[1],3)), 5611.054);
    expect_equal(suppressWarnings(round(forecast(testModel, h=18, holdout=FALSE)$forecast[18],3)), 6181.582);
})
