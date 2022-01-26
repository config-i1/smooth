context("Tests for ces() function");

# Basic CES selection
testModel <- auto.ces(BJsales, silent=TRUE);
test_that("Test CES selection on BJsales", {
    expect_match(testModel$model, "n");
})

# Reuse previous CES
test_that("Test on BJsales, predefined CES", {
    expect_equal(ces(BJsales, model=testModel, silent=TRUE)$cf, testModel$cf);
})

# Test trace cost function for CES
testModel <- ces(AirPassengers, seasonality="f", h=18, holdout=TRUE, silent=TRUE, interval=TRUE)
test_that("Test AICc of CES based on MSTFE on AirPassengers", {
    expect_equal(as.numeric(round(AICc(testModel),2)), as.numeric(round(testModel$ICs["AICc"],2)));
})

# Test how different passed values are accepted by CES
test_that("Test initials, a and b of CES on AirPassengers", {
    expect_equal(ces(AirPassengers, seasonality="f", initial=testModel$initial, silent=TRUE)$initial, testModel$initial);
    expect_equal(ces(AirPassengers, seasonality="f", a=testModel$a, silent=TRUE)$a, testModel$a);
    expect_equal(ces(AirPassengers, seasonality="f", b=testModel$b, silent=TRUE)$b, testModel$b);
})

# Test selection of exogenous with CES
test_that("Select exogenous variables for CESX on BJsales with selection", {
    skip_on_cran()
    x <- BJsales.lead;
    y <- BJsales;
    testModel <- ces(y, h=18, holdout=TRUE, xreg=xregExpander(x), silent=TRUE, xregDo="select")
    expect_equal(suppressWarnings(ncol(testModel$xreg)),1);
})
