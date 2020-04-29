context("Tests for simulate() functions");

#### ETS ####
testData <- sim.es("MNN", frequency=12, bounds="a", obs=100);
test_that("ETS(MNN) simulated with admissible bounds", {
    expect_match(testData$model, "MNN");
})

testData <- sim.es("AAdM", frequency=12, phi=0.9, obs=120);
test_that("ETS(AAdM) simulated with phi=0.9", {
    expect_match(testData$model, "AAdM");
})

testData <- sim.es("MNN", frequency=12, obs=120, nsim=100, probability=0.2);
test_that("iETS(MNN) simulated with probability=0.2 and nsim=100", {
    expect_match(testData$model, "MNN");
})

testModel <- es(Mcomp::M3$N1984$x, "ANA", h=18, silent=TRUE);
test_that("ETS(ANA) simulated from estimated model", {
    expect_match(simulate(testModel,nsim=10,seed=5,obs=100)$model, "ANA");
})

#### SSARIMA ####
testModel <- auto.ssarima(Mcomp::M3$N1234$x, h=8, silent=TRUE);
test_that("ARIMA(0,1,3) with drift simulated from estimated model", {
    expect_match(simulate(testModel,nsim=10,seed=5,obs=100)$model, "(0,1,3)");
})

test_that("ARIMA(0,1,1) with intermittent data", {
    expect_match(sim.ssarima(nsim=10,obs=100,probability=0.2)$model, "iARIMA");
})

#### CES ####
testModel <- auto.ces(Mcomp::M3$N1234$x, h=8, silent=TRUE);
test_that("CES(n) simulated from estimated model", {
    expect_match(simulate(testModel,nsim=10,seed=5,obs=100)$model, "(n)");
})

test_that("CES(s) with some random parameters", {
    expect_match(sim.ces(seasonality="s",frequency=4,nsim=1,obs=100)$model, "(s)");
})

test_that("CES(p) with some random A parameter and fixed b=0.1 ", {
    expect_equal(sim.ces(seasonality="p",frequency=4,b=0.1,nsim=1,obs=100)$b[1], 0.1);
})

test_that("CES(f) with intermittent data", {
    expect_match(sim.ces(seasonality="f",frequency=12,nsim=10,obs=100,probability=0.2)$model, "iCES");
})

#### GUM ####
testModel <- gum(Mcomp::M3$N1234$x, orders=1, lags=1, h=8, silent=TRUE);
test_that("GUM(1[1]) simulated from estimated model", {
    expect_match(simulate(testModel,nsim=10,seed=5,obs=100)$model, "GUM");
})

test_that("GUM(1[1]) with intermittent data", {
    expect_match(sim.gum(nsim=10,obs=100,probability=0.2)$model, "iGUM");
})

#### SMA ####
test_that("SMA(10) with intermittent data", {
    expect_match(sim.sma(10,nsim=10,obs=100,probability=0.2)$model, "iSMA");
})

#### VES ####
Y <- cbind(Mcomp::M3$N2570$x,Mcomp::M3$N2571$x);
testModel <- ves(Y,"AAN", silent=TRUE)
test_that("VES based on pre-estimated model", {
    expect_match(simulate(testModel,nsim=10,obs=100)$model, "VES");
})
