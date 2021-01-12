context("Tests for ves() function")

Y <- cbind(Mcomp::M3$N2570$x,Mcomp::M3$N2571$x)

# Basic VES check
testModel <- suppressWarnings(ves(Y,"MAdM", silent=TRUE))
test_that("Test VES(MMdM)", {
    expect_match(testModel$model, "MMdM")
})

# Reuse previous VES
test_that("Reuse VES", {
    expect_equal(ves(Y, model=testModel, silent=TRUE)$persistence, testModel$persistence)
})

# Test VES with individual seasonality and persistence
test_that("Test VES with individual seasonality and persistence", {
    skip_on_cran()
    testModel <- ves(Y,"MMdM", initialSeason="i", persistence="i", silent=TRUE)
    expect_equal(length(coefficients(testModel)), 35)
})

# Test VES with grouped initials and dependent persistence
test_that("Test VES with grouped initials and dependent persistence", {
    skip_on_cran()
    testModel <- ves(Y,"AAN", initial="c", persistence="d", silent=TRUE)
    expect_equal(length(coefficients(testModel)), 10)
})

# Test VES with a trace cost function
test_that("Test VES with a trace cost function", {
    skip_on_cran()
    testModel <- ves(Y,"AAN", loss="t", silent=TRUE)
    expect_match(testModel$loss, "trace")
})

# Test VES with a dependent transition and independent interval
test_that("Test VES with a dependent transition and independent interval", {
    skip_on_cran()
    testModel <- ves(Y,"AAN", transition="d", interval="i", silent=TRUE)
    expect_false(isTRUE(all.equal(testModel$transition[1,4], 0)))
    expect_equal(dim(testModel$PI),c(10,4))
})
