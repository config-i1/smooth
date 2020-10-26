context("Tests for gum() function");

# Basic GUM selection
testModel <- gum(Mcomp::M3$N1234$x, orders=c(2,1),lags=c(1,4), silent=TRUE);
test_that("Test if GUM worked on N1234$x", {
    expect_equal(testModel$model, "GUM(2[1],1[4])");
})

# Reuse previous GUM
test_that("Reuse previous GUM on N1234$x", {
    expect_equal(gum(Mcomp::M3$N1234$x, model=testModel, silent=TRUE)$cf, testModel$cf);
})

# Test some crazy order of GUM
testModel <- gum(Mcomp::M3$N1234$x, orders=c(1,1,1), lags=c(1,3,5), h=18, holdout=TRUE, initial="o", silent=TRUE, interval=TRUE)
test_that("Test if crazy order GUM was estimated on N1234$x", {
    expect_equal(testModel$model, "GUM(1[1],1[3],1[5])");
})

# Test how different passed values are accepted by GUM
test_that("Test initials, measurement, transition and persistence of GUM on N2568$x", {
    expect_equal(gum(Mcomp::M3$N2568$x, orders=c(1,1,1), lags=c(1,3,5), initial=testModel$initial, silent=TRUE)$initial, testModel$initial);
    expect_equal(gum(Mcomp::M3$N2568$x, orders=c(1,1,1), lags=c(1,3,5), measurement=testModel$measurement, silent=TRUE)$measurement, testModel$measurement);
    expect_equal(gum(Mcomp::M3$N2568$x, orders=c(1,1,1), lags=c(1,3,5), transition=testModel$transition, silent=TRUE)$transition, testModel$transition);
    expect_equal(gum(Mcomp::M3$N2568$x, orders=c(1,1,1), lags=c(1,3,5), persistence=testModel$persistence, silent=TRUE)$persistence, testModel$persistence);
})

# Test selection of exogenous with GUM
x <- cbind(c(rep(0,25),1,rep(0,43)),c(rep(0,10),1,rep(0,58)));
y <- ts(c(Mcomp::M3$N1457$x,Mcomp::M3$N1457$xx),frequency=12);
testModel <- gum(y, h=18, holdout=TRUE, xreg=xregExpander(x), silent=TRUE, xregDo="select")
test_that("Select exogenous variables for GUMX on N1457", {
    expect_match(errorType(testModel),"A");
})

# Use automatic GUM
test_that("Use automatic GUM on N1234$x", {
    expect_equal(auto.gum(Mcomp::M3$N1234, silent=TRUE, bounds="r")$model, "GUM(1[1],1[3])");
})
