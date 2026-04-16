context("ADAM baseline regression tests")

# ── helpers ───────────────────────────────────────────────────────────────────
.ref_path <- testthat::test_path("adam_baseline.rds")

.snapshot <- function(m) {
    list(
        loss_value = round(m$lossValue, 5),
        loglik     = round(as.numeric(logLik(m)), 4),
        coef       = round(coef(m), 5)
    )
}

if (!file.exists(.ref_path)) {
    message("adam_baseline.rds not found — generating reference on first run.")
    .generate_ref <- TRUE
    .ref <- list()
} else {
    .generate_ref <- FALSE
    .ref <- readRDS(.ref_path)
}

.check <- function(tag, m) {
    snap <- .snapshot(m)
    if (.generate_ref) {
        .ref[[tag]] <<- snap
    } else {
        r <- .ref[[tag]]
        test_that(paste("baseline:", tag, "— lossValue"), {
            skip_on_cran()
            expect_equal(snap$loss_value, r$loss_value, tolerance=1e-4)
        })
        test_that(paste("baseline:", tag, "— logLik"), {
            skip_on_cran()
            expect_equal(snap$loglik, r$loglik, tolerance=1e-3)
        })
        test_that(paste("baseline:", tag, "— coef"), {
            skip_on_cran()
            expect_equal(snap$coef, r$coef, tolerance=1e-4)
        })
    }
}

# ── data ──────────────────────────────────────────────────────────────────────
set.seed(1)
xreg <- matrix(rnorm(length(BJsales) * 2), ncol=2,
               dimnames=list(NULL, c("x1","x2")))
xreg_air <- cbind(
    trend  = 1:length(AirPassengers),
    season = rep(1:12, length.out=length(AirPassengers))
)
BJ_df  <- data.frame(BJsales=as.numeric(BJsales), x1=xreg[,1], x2=xreg[,2])
air_df <- data.frame(AirPassengers=as.numeric(AirPassengers),
                     trend=xreg_air[,1], season=xreg_air[,2])

set.seed(1)
occModel  <- sim.oes("MNN", 120, frequency=12, occurrence="general",
                     persistence=0.01, initial=2, initialB=1)
occData   <- sim.es("MNN", 120, frequency=12,
                    probability=occModel$probability, persistence=0.1)$data

# ── ETS variants — BJsales ────────────────────────────────────────────────────
set.seed(1); .check("ETS_ANN",    adam(BJsales, "ANN"))
set.seed(1); .check("ETS_MNN",    adam(BJsales, "MNN"))
set.seed(1); .check("ETS_AAN",    adam(BJsales, "AAN"))
set.seed(1); .check("ETS_MAN",    adam(BJsales, "MAN"))
set.seed(1); .check("ETS_AAdN",   adam(BJsales, "AAdN"))
set.seed(1); .check("ETS_MAdN",   adam(BJsales, "MAdN"))
set.seed(1); .check("ETS_ZZZ_BJ", adam(BJsales, "ZZZ"))

# ── ETS variants — AirPassengers ─────────────────────────────────────────────
set.seed(1); .check("ETS_AAA_air",   adam(AirPassengers, "AAA",  lags=12))
set.seed(1); .check("ETS_MAM_air",   adam(AirPassengers, "MAM",  lags=12))
set.seed(1); .check("ETS_MAdM_air",  adam(AirPassengers, "MAdM", lags=12))
set.seed(1); .check("ETS_ZXZ_air",   adam(AirPassengers, "ZXZ",  lags=12))
set.seed(1); .check("ETS_ZZZ_air",   adam(AirPassengers, "ZZZ",  lags=12))

# ── Initial methods ───────────────────────────────────────────────────────────
set.seed(1); .check("ETS_AAN_optimal",   adam(BJsales, "AAN", initial="optimal"))
set.seed(1); .check("ETS_AAN_twostage",  adam(BJsales, "AAN", initial="two-stage"))
set.seed(1); .check("ETS_AAN_complete",  adam(BJsales, "AAN", initial="complete"))

# ── Loss functions ────────────────────────────────────────────────────────────
set.seed(1); .check("LOSS_MSE",   adam(BJsales, "MAN", loss="MSE"))
set.seed(1); .check("LOSS_MAE",   adam(BJsales, "MAN", loss="MAE"))
set.seed(1); .check("LOSS_HAM",   adam(BJsales, "MAN", loss="HAM"))
set.seed(1); .check("LOSS_MSEh",  adam(BJsales, "MAN", loss="MSEh",  h=12))
set.seed(1); .check("LOSS_TMSE",  adam(BJsales, "MAN", loss="TMSE",  h=12))
set.seed(1); .check("LOSS_GTMSE", adam(BJsales, "MAN", loss="GTMSE", h=12))
set.seed(1); .check("LOSS_GPL",   adam(BJsales, "MAN", loss="GPL",   h=12))
set.seed(1); .check("LOSS_LASSO", adam(BJsales, "MAN", loss="LASSO", lambda=0.1))
set.seed(1); .check("LOSS_RIDGE", adam(BJsales, "MAN", loss="RIDGE", lambda=0.1))

# ── Distributions ─────────────────────────────────────────────────────────────
set.seed(1); .check("DIST_dnorm",    adam(BJsales,       "AAN",  distribution="dnorm"))
set.seed(1); .check("DIST_dlaplace", adam(BJsales,       "AAN",  distribution="dlaplace"))
set.seed(1); .check("DIST_ds",       adam(BJsales,       "AAN",  distribution="ds"))
set.seed(1); .check("DIST_dgnorm",   adam(BJsales,       "MAN",  distribution="dgnorm"))
set.seed(1); .check("DIST_dlnorm",   adam(AirPassengers, "MAM",  lags=12, distribution="dlnorm"))
set.seed(1); .check("DIST_dgamma",   adam(AirPassengers, "MAM",  lags=12, distribution="dgamma"))
set.seed(1); .check("DIST_dinvgauss",adam(AirPassengers, "MAM",  lags=12, distribution="dinvgauss"))

# ── ARIMA ─────────────────────────────────────────────────────────────────────
set.seed(1); .check("ARIMA_022",
    adam(BJsales, "NNN", orders=c(0,2,2)))
set.seed(1); .check("ARIMA_111",
    adam(BJsales, "NNN", orders=c(1,1,1)))
set.seed(1); .check("SARIMA_air",
    adam(AirPassengers, "NNN", lags=12,
         orders=list(ar=c(0,1), i=c(1,1), ma=c(1,1))))
set.seed(1); .check("ETS_ARIMA",
    adam(BJsales, "AAN", orders=c(1,0,1)))
set.seed(1); .check("ETS_ARIMA_seasonal",
    adam(AirPassengers, "AAA", lags=12,
         orders=list(ar=c(1,1), i=c(0,0), ma=c(0,0))))
set.seed(1); .check("ARIMA_constant",
    adam(BJsales, "NNN", orders=c(0,1,1), constant=TRUE))

# ── Regression (ETSX / ARIMAX) ────────────────────────────────────────────────
set.seed(1); .check("ETSX_use",
    adam(BJ_df, "AAN", formula=BJsales~., lags=1))
set.seed(1); .check("ETSX_select",
    adam(BJ_df, "AAN", formula=BJsales~., lags=1, regressors="select"))
set.seed(1); .check("ETSX_adapt",
    adam(BJ_df, "AAN", formula=BJsales~., lags=1, regressors="adapt"))
set.seed(1); .check("ARIMAX",
    adam(BJ_df, "NNN", orders=c(0,1,1), formula=BJsales~., lags=1))
set.seed(1); .check("ETSX_air",
    adam(air_df, "ZXZ", lags=12, formula=AirPassengers~.))

# ── Outlier handling ──────────────────────────────────────────────────────────
set.seed(1); .check("OUTLIER_use",     adam(BJsales,       "AAN", outliers="use"))
set.seed(1); .check("OUTLIER_select",  adam(BJsales,       "AAN", outliers="select"))
set.seed(1); .check("OUTLIER_air_use", adam(AirPassengers, "ZXZ", lags=12, outliers="use"))

# ── Holdout ───────────────────────────────────────────────────────────────────
set.seed(1); .check("HOLDOUT_BJ",  adam(BJsales,       "AAN", h=10, holdout=TRUE))
set.seed(1); .check("HOLDOUT_air", adam(AirPassengers, "ZXZ", lags=12, h=12, holdout=TRUE))

# ── Fixed / partially-fixed parameters ───────────────────────────────────────
set.seed(1); .check("FIXED_alpha",
    adam(BJsales, "AAN", persistence=list(alpha=0.1)))
set.seed(1); .check("FIXED_phi",
    adam(BJsales, "AAdN", phi=0.95))
set.seed(1); .check("FIXED_initial_level",
    adam(BJsales, "ANN", initial=list(level=200)))
set.seed(1); .check("FIXED_initial_level_trend",
    adam(BJsales, "AAN", initial=list(level=200, trend=1)))
set.seed(1); .check("FIXED_initial_seasonal",
    adam(AirPassengers, "AAA", lags=12,
         initial=list(level=300, trend=1, seasonal=rep(0,12))))
set.seed(1); .check("FIXED_initial_alpha",
    adam(BJsales, "AAN", initial=list(level=200), persistence=list(alpha=0.2)))

# ── Occurrence models ─────────────────────────────────────────────────────────
set.seed(1); .check("OCC_fixed",   adam(occData, "MNN", lags=12, occurrence="fixed"))
set.seed(1); .check("OCC_general", adam(occData, "MNN", lags=12, occurrence="general"))
set.seed(1); .check("OCC_direct",  adam(occData, "MNN", lags=12, occurrence="direct"))
set.seed(1); .check("OCC_auto",    adam(occData, "ZZN", lags=12, occurrence="auto"))

# ── Save reference on first run ───────────────────────────────────────────────
if (.generate_ref) {
    saveRDS(.ref, .ref_path)
    message("Reference saved to ", .ref_path)

    # Emit trivial passing tests so the first run is not silent
    for (.tag in names(.ref)) {
        local({
            tag <- .tag
            test_that(paste("baseline: generated reference for", tag), {
                skip_on_cran()
                expect_true(!is.null(.ref[[tag]]))
            })
        })
    }
}
