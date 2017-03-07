context("Tests for es() function");

# Basic ETS selection
testModel <- es(Mcomp::M3$N1234$x, silent=TRUE);
test_that("Test ETS selection on N1234$x", {
    expect_match(testModel$model, "MAN");
})

test_that("Test damped-trend ETS on N1234$x", {
    expect_equal(round(es(Mcomp::M3$N1234$x,model="AAdN", silent=TRUE)$phi,2), 0.96);
})

# Reuse previous ETS
test_that("Test on N1234$x, predefined ETS", {
    expect_equal(es(Mcomp::M3$N1234$x, model=testModel, silent=TRUE)$cf, testModel$cf);
})

# Test combinations of ETS
testModel <- es(Mcomp::M3$N2568$x, "CCC", silent=TRUE, ic="BIC");
test_that("Test ETS(CCC) with BIC on N2568$x", {
    expect_equal(testModel$s2, mean(testModel$residuals^2));
})

# Test model selection of non-multiplicative trend ETS
testModel <- es(Mcomp::M3$N2568$x, "MXM", silent=TRUE, ic="AIC");
test_that("Test ETS(MXM) with AIC on N2568$x", {
    expect_match(testModel$model, "MAM");
})

# Test trace cost function for ETS
testModel <- es(Mcomp::M3$N2568$x, model="MAdM", h=18, holdout=TRUE, silent=TRUE, intervals=TRUE)
test_that("Test AIC of ETS on N2568$x", {
    expect_equal(as.numeric(round(AIC(testModel),2)), as.numeric(round(testModel$ICs["AIC"],2)));
})

# Test how different passed values are accepted by ETS
test_that("Test initials, initialSeason and persistence of ETS on N2568$x", {
    expect_equal(es(Mcomp::M3$N2568$x, model="MAdM", initial=testModel$initial, silent=TRUE)$initial, testModel$initial);
    expect_equal(es(Mcomp::M3$N2568$x, model="MAdM", persistence=testModel$persistence, silent=TRUE)$persistence, testModel$persistence);
    expect_equal(es(Mcomp::M3$N2568$x, model="MAdM", initialSeason=testModel$initialSeason, silent=TRUE)$initialSeason, testModel$initialSeason);
    expect_equal(es(Mcomp::M3$N2568$x, model="MAdM", phi=testModel$phi, silent=TRUE)$phi, testModel$phi);
})

# Test exogenous (normal + updateX) with ETS
x <- cbind(c(rep(0,25),1,rep(0,43)),c(rep(0,10),1,rep(0,58)));
y <- ts(c(Mcomp::M3$N1457$x,Mcomp::M3$N1457$xx),frequency=12);
testModel <- es(y, h=18, holdout=TRUE, xreg=x, updateX=TRUE, silent=TRUE, intervals="np")
test_that("Check exogenous variables for ETS on N1457", {
    expect_equal(suppressWarnings(es(y, h=18, holdout=TRUE, xreg=x, cfType="aMSTFE", silent=TRUE)$model), testModel$model);
    expect_equal(suppressWarnings(forecast(testModel, h=18, holdout=FALSE)$method), testModel$model);
})

# Test selection of exogenous with ETS
testModel <- es(y, h=18, holdout=TRUE, xreg=xregExpander(x), silent=TRUE, xregDo="select")
test_that("Select exogenous variables for ETS on N1457 with selection", {
    expect_equal(ncol(testModel$xreg),3);
})

# Test combination of ETS with exogenous selection
testModel <- es(y, "CCC", h=18, holdout=TRUE, xreg=x, silent=TRUE, xregDo="select")
test_that("Select exogenous variables for ETSX combined on N1457", {
    expect_match(testModel$model, "ETSX");
})

# iETS test
x <- c(0,1,2,0,0,0,1,0,0,1,0,0,0,2,0,0,0,1,0,0);
testModel <- es(x, "MNN", intermittent="a", silent=TRUE, ic="AIC");
test_that("Test ETS selection on N1234$x", {
    expect_match(testModel$model, "iETS");
})
