context("Tests for es() function");

# Basic ETS selection
testModel <- es(Mcomp::M3$N1234$x, silent=TRUE);
test_that("Test ETS selection on N1234$x", {
    expect_match(testModel$model, "MAN");
})

# Reuse previous ETS
test_that("Test on N1234$x, predefined ETS", {
    expect_equal(round(es(Mcomp::M3$N1234$x, model=testModel, silent=TRUE)$cf,2), 4676.66);
})

# Test combinations of ETS
testModel <- es(Mcomp::M3$N2568$x, "CCC", silent=TRUE);
test_that("Test ETS(CCC) on N2568$x", {
    expect_equal(round(testModel$s2,5), 0.00418);
})

# Test trace cost function for ETS
testModel <- es(Mcomp::M3$N2568$x, model="MAdM", h=18, holdout=TRUE, cfType="MSTFE", silent=TRUE)
test_that("Test AIC of ETS based on MSTFE on N2568$x", {
    expect_equal(as.vector(round(AIC(testModel),2)), 26306.84);
})

# Test how different passed values are accepted by ETS
test_that("Test initials, initialSeason and persistence of ETS on N2568$x", {
    expect_equal(as.vector(round(es(Mcomp::M3$N2568$x, model="MAdM", initial=testModel$initial, silent=TRUE)$initial,5)), c(4711.74455,51.20892));
    expect_equal(as.vector(round(es(Mcomp::M3$N2568$x, model="MAdM", persistence=testModel$persistence, silent=TRUE)$persistence,5)), c(0.05825,0.00091,0.58992));
    expect_equal(as.vector(round(es(Mcomp::M3$N2568$x, model="MAdM", initialSeason=testModel$initialSeason, silent=TRUE)$initialSeason,3)), c(1.219,0.811,0.898,1.379,0.678,1.293,0.870,0.941,1.256,0.887,0.888,1.151));
    expect_equal(as.vector(round(es(Mcomp::M3$N2568$x, model="MAdM", phi=testModel$phi, silent=TRUE)$phi,3)), 0.995);
})

# Test exogenous (normal + updateX) with ETS
x <- cbind(c(rep(0,25),1,rep(0,43)),c(rep(0,10),1,rep(0,58)));
y <- ts(c(Mcomp::M3$N1457$x,Mcomp::M3$N1457$xx),frequency=12);
testModel <- es(y, h=18, holdout=TRUE, xreg=x, updateX=TRUE, silent=TRUE)
test_that("Check exogenous variables for ETS on N1457", {
    expect_equal(suppressWarnings(round(es(y, h=18, holdout=TRUE, xreg=x, cfType="aMSTFE", silent=TRUE)$forecast[1],3)), 5776.014);
    expect_equal(suppressWarnings(round(forecast(testModel, h=18, holdout=FALSE)$forecast[18],3)), 5721.698);
})
