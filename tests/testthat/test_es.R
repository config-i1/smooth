context("Tests for es() function");

# Basic model selection
testModel <- es(Mcomp::M3$N1234$x, silent=TRUE);
test_that("Test model selection on N1234$x", {
    expect_match(testModel$model, "MAN");
})

# Reuse previous model
test_that("Test on N1234$x, predefined model", {
    expect_equal(round(es(Mcomp::M3$N1234$x, model=testModel)$cf,2), 4676.66);
})

# Test combinations
testModel <- es(Mcomp::M3$N2568$x, "CCC", silent=TRUE);
test_that("Test ETS(CCC) on N2568$x", {
    expect_equal(round(testModel$s2,5), 0.00418);
})

# Test combinations
testModel <- es(Mcomp::M3$N2568$x,model="MAdM",h=18,holdout=TRUE,cfType="MSTFE", silent=TRUE)
test_that("Test AIC of model based on MSTFE on N2568$x", {
    expect_equal(as.vector(round(AIC(testModel),2)), 26306.84);
})

# Test how different passed values are taken by the function
test_that("Test initials, initialSeason and persistence on N2568$x", {
    expect_equal(as.vector(round(es(Mcomp::M3$N2568$x, model="MAdM", initial=testModel$initial)$initial,5)), c(4711.74455,51.20892));
    expect_equal(as.vector(round(es(Mcomp::M3$N2568$x, model="MAdM", persistence=testModel$persistence)$persistence,5)), c(0.05825,0.00091,0.58992));
    expect_equal(as.vector(round(es(Mcomp::M3$N2568$x, model="MAdM", initialSeason=testModel$initialSeason)$initialSeason,3)), c(1.219,0.811,0.898,1.379,0.678,1.293,0.870,0.941,1.256,0.887,0.888,1.151));
    expect_equal(as.vector(round(es(Mcomp::M3$N2568$x, model="MAdM", phi=testModel$phi)$phi,3)), 0.995);
})

# Test exogenous (normal + updateX)
x <- cbind(c(rep(0,25),1,rep(0,43)),c(rep(0,10),1,rep(0,58)));
y <- ts(c(Mcomp::M3$N1457$x,Mcomp::M3$N1457$xx),frequency=12);
testModel <- es(y,h=18,holdout=TRUE,xreg=x,updateX=TRUE, silent=TRUE)
test_that("Check exogenous variables on N1457", {
    expect_equal(suppressWarnings(round(es(y,h=18,holdout=TRUE,xreg=x,cfType="aMSTFE", silent=TRUE)$forecast[1],3)), 5776.014);
    expect_equal(suppressWarnings(round(forecast(testModel,h=18,holdout=FALSE)$forecast[18],3)), 5721.698);
})
