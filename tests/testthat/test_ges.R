context("Tests for ges() function");

# Basic GES selection
testModel <- ges(Mcomp::M3$N1234$x, orders=c(2,1),lags=c(1,4), silent=TRUE);
test_that("Test if GES worked on N1234$x", {
    expect_equal(testModel$model, "GES(2[1],1[4])");
})

# Reuse previous GES
test_that("Reuse previous GES on N1234$x", {
    expect_equal(ges(Mcomp::M3$N1234$x, model=testModel, silent=TRUE)$cf, testModel$cf);
})

# Test some crazy order of GES
testModel <- ges(Mcomp::M3$N1234$x, orders=c(1,1,1), lags=c(1,3,5), h=18, holdout=TRUE, initial="o", silent=TRUE, intervals=TRUE)
test_that("Test if crazy order GES was estimated on N1234$x", {
    expect_equal(testModel$model, "GES(1[1],1[3],1[5])");
})

# Test how different passed values are accepted by GES
test_that("Test initials, measurement, transition and persistence of GES on N2568$x", {
    expect_equal(ges(Mcomp::M3$N2568$x, orders=c(1,1,1), lags=c(1,3,5), initial=testModel$initial, silent=TRUE)$initial, testModel$initial);
    expect_equal(ges(Mcomp::M3$N2568$x, orders=c(1,1,1), lags=c(1,3,5), measurement=testModel$measurement, silent=TRUE)$measurement, testModel$measurement);
    expect_equal(ges(Mcomp::M3$N2568$x, orders=c(1,1,1), lags=c(1,3,5), transition=testModel$transition, silent=TRUE)$transition, testModel$transition);
    expect_equal(ges(Mcomp::M3$N2568$x, orders=c(1,1,1), lags=c(1,3,5), persistence=testModel$persistence, silent=TRUE)$persistence, testModel$persistence);
})

# Test exogenous (normal + updateX) with GES
x <- cbind(c(rep(0,25),1,rep(0,43)),c(rep(0,10),1,rep(0,58)));
y <- ts(c(Mcomp::M3$N1457$x,Mcomp::M3$N1457$xx),frequency=12);
testModel <- ges(y, h=18, holdout=TRUE, xreg=x, updateX=TRUE, silent=TRUE, cfType="aMSTFE", intervals=TRUE, intervalsType="a")
test_that("Check exogenous variables for GES on N1457", {
    expect_equal(suppressWarnings(ges(y, h=18, holdout=TRUE, xreg=x, silent=TRUE)$model), testModel$model);
    expect_equal(suppressWarnings(forecast(testModel, h=18, holdout=FALSE)$model), testModel$model);
})
