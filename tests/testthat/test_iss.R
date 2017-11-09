context("Tests for iss() function");

# ISS with fixed probability
testModel <- iss(rpois(100,0.2), intermittent="f");
test_that("Test ISS with fixed probability", {
    expect_equal(testModel$intermittent, "f");
})

# ISS with Croston's probability
testModel <- iss(rpois(100,0.2), intermittent="i");
test_that("Test ISS with Croston's probability", {
    expect_equal(testModel$intermittent, "i");
})

# ISS with TSB probability
testModel <- iss(rpois(100,0.2), intermittent="p");
test_that("Test ISS with TSB probability", {
    expect_equal(testModel$intermittent, "p");
})

# ISS with Croston's probability and ETS(MMN)
testModel <- iss(rpois(100,0.2), intermittent="i", model="MMN");
test_that("Test ISS with Croston's probability and underlying ETS(MMN)", {
    expect_equal(length(testModel$persistence), 2);
})
