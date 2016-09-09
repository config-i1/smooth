context("Tests for simulate() functions");

# ISS with fixed probability
testData <- sim.es("MNN", 12, bounds="a", obs=100, silent=TRUE);
test_that("ETS(MNN) simulated with admissible bounds", {
    expect_match(testData$model, "MNN");
})

testData <- sim.es("AAdM", 12, phi=0.9, obs=120);
test_that("ETS(AAdM) simulated with phi=0.9", {
    expect_match(testData$model, "AAdM");
})

testData <- sim.es("MNN", 12, obs=120, nsim=100, iprob=0.2);
test_that("iETS(MNN) simulated with iprob=0.2 and nsim=100", {
    expect_match(testData$model, "MNN");
})

testModel <- es(Mcomp::M3$N1984$x, "ANA", h=18, silent=TRUE)
test_that("ETS(ANA) simulated from estimated model", {
    expect_match(simulate(testModel,nsim=10,seed=5,obs=100)$model, "ANA");
})
