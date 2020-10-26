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
testModel <- ces(Mcomp::M3$N2568$x, seasonality="f", h=18, holdout=TRUE, silent=TRUE, interval=TRUE)
test_that("Test AICc of CES based on MSTFE on N2568$x", {
    expect_equal(as.numeric(round(AICc(testModel),2)), as.numeric(round(testModel$ICs["AICc"],2)));
})

# Test how different passed values are accepted by CES
test_that("Test initials, a and b of CES on N2568$x", {
    expect_equal(ces(Mcomp::M3$N2568$x, seasonality="f", initial=testModel$initial, silent=TRUE)$initial, testModel$initial);
    expect_equal(ces(Mcomp::M3$N2568$x, seasonality="f", a=testModel$a, silent=TRUE)$a, testModel$a);
    expect_equal(ces(Mcomp::M3$N2568$x, seasonality="f", b=testModel$b, silent=TRUE)$b, testModel$b);
})

# Test selection of exogenous with CES
x <- cbind(c(rep(0,25),1,rep(0,43)),c(rep(0,10),1,rep(0,58)));
y <- ts(c(Mcomp::M3$N1457$x,Mcomp::M3$N1457$xx),frequency=12);
testModel <- ces(y, h=18, holdout=TRUE, xreg=xregExpander(x), silent=TRUE, xregDo="select")
test_that("Select exogenous variables for CESX on N1457 with selection", {
    expect_equal(suppressWarnings(ncol(testModel$xreg)),3);
})
