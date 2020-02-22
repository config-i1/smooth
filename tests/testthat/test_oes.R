context("Tests for oes() function");

# ISS with fixed probability
testModel <- oes(rpois(100,0.2), occurrence="f");
test_that("Test oes with fixed probability", {
    expect_equal(testModel$occurrence, "fixed");
})

# oes with Inverse odds ratio probability
testModel <- oes(rpois(100,0.2), occurrence="i");
test_that("Test oes with Iverse odds ratio probability", {
    expect_equal(testModel$occurrence, "inverse-odds-ratio");
})

# oes with odds ratio probability
testModel <- oes(rpois(100,0.2), occurrence="o");
test_that("Test oes with Odds ratio probability", {
    expect_equal(testModel$occurrence, "odds-ratio");
})

# oes with odds ratio probability and ETS(MMN)
testModel <- oes(rpois(100,0.2), occurrence="o", model="MMN");
test_that("Test oes with Odds ratio probability and ETS(MMN)", {
    expect_equal(length(testModel$persistence), 2);
})

# oes with automatically selected type of model and ETS(MMN)
testModel <- oes(rpois(100,0.2), occurrence="a");
test_that("Test oes with auto probability selected", {
    expect_equal(length(testModel$persistence), 1);
})
