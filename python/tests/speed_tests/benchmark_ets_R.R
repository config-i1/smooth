# ETS Benchmark Comparison: R smooth vs forecast::ets
#
# This script benchmarks ETS model performance on M3, Tourism, and M1 datasets
# using R packages. Run this to compare with Python benchmark results.
#
# Usage:
#   Rscript benchmark_ets_R.R
#   Rscript benchmark_ets_R.R --n_series 10
#   Rscript benchmark_ets_R.R --dataset M3 --subset Monthly
#
# Requirements:
#   install.packages(c("smooth", "Mcomp", "Tcomp", "forecast"))

library(smooth)
library(Mcomp)
library(forecast)

# Try to load Tcomp for Tourism data (may not be installed)
has_tcomp <- suppressWarnings(require(Tcomp, quietly = TRUE))

# =============================================================================
# Error Metrics
# =============================================================================

mae <- function(y_true, y_pred) {
    mean(abs(y_true - y_pred), na.rm = TRUE)
}

rmse <- function(y_true, y_pred) {
    sqrt(mean((y_true - y_pred)^2, na.rm = TRUE))
}

smape <- function(y_true, y_pred) {
    denominator <- abs(y_true) + abs(y_pred)
    denominator[denominator == 0] <- 1
    200 * mean(abs(y_true - y_pred) / denominator, na.rm = TRUE)
}

mase <- function(y_true, y_pred, y_train, seasonality = 1) {
    if (seasonality == 1) {
        naive_errors <- abs(diff(y_train))
    } else {
        n <- length(y_train)
        naive_errors <- abs(y_train[(seasonality+1):n] - y_train[1:(n-seasonality)])
    }
    scale <- mean(naive_errors, na.rm = TRUE)
    if (scale == 0 || is.na(scale)) scale <- 1
    mean(abs(y_true - y_pred), na.rm = TRUE) / scale
}

# =============================================================================
# Benchmark Functions
# =============================================================================

benchmark_smooth_adam <- function(data, initial = "back") {
    # Benchmark smooth::adam
    start_time <- Sys.time()

    result <- tryCatch({
        fit <- adam(data$x, model = "ZZZ", initial = initial)
        fc <- forecast(fit, h = data$h)
        list(
            forecast = as.vector(fc$mean),
            time = as.numeric(Sys.time() - start_time, units = "secs"),
            model_type = fit$model,
            success = TRUE
        )
    }, error = function(e) {
        list(
            forecast = rep(NA, data$h),
            time = as.numeric(Sys.time() - start_time, units = "secs"),
            model_type = "ERROR",
            success = FALSE,
            error = as.character(e)
        )
    })

    return(result)
}

benchmark_forecast_ets <- function(data) {
    # Benchmark forecast::ets
    start_time <- Sys.time()

    result <- tryCatch({
        fit <- ets(data$x)
        fc <- forecast(fit, h = data$h)
        list(
            forecast = as.vector(fc$mean),
            time = as.numeric(Sys.time() - start_time, units = "secs"),
            model_type = fit$method,
            success = TRUE
        )
    }, error = function(e) {
        list(
            forecast = rep(NA, data$h),
            time = as.numeric(Sys.time() - start_time, units = "secs"),
            model_type = "ERROR",
            success = FALSE,
            error = as.character(e)
        )
    })

    return(result)
}

benchmark_single_series <- function(data, methods = c("smooth_back", "smooth_opt", "ets")) {
    # Benchmark a single Mcomp series across all methods

    y_train <- as.vector(data$x)
    y_test <- as.vector(data$xx)
    h <- data$h
    seasonality <- frequency(data$x)
    if (is.null(seasonality) || seasonality < 1) seasonality <- 1

    results <- list()

    # smooth ADAM - backcasting
    if ("smooth_back" %in% methods) {
        res <- benchmark_smooth_adam(data, initial = "back")
        if (res$success) {
            fc_len <- min(length(res$forecast), length(y_test))
            res$mae <- mae(y_test[1:fc_len], res$forecast[1:fc_len])
            res$rmse <- rmse(y_test[1:fc_len], res$forecast[1:fc_len])
            res$smape <- smape(y_test[1:fc_len], res$forecast[1:fc_len])
            res$mase <- mase(y_test[1:fc_len], res$forecast[1:fc_len], y_train, seasonality)
        }
        results$smooth_back <- res
    }

    # smooth ADAM - optimal
    if ("smooth_opt" %in% methods) {
        res <- benchmark_smooth_adam(data, initial = "opt")
        if (res$success) {
            fc_len <- min(length(res$forecast), length(y_test))
            res$mae <- mae(y_test[1:fc_len], res$forecast[1:fc_len])
            res$rmse <- rmse(y_test[1:fc_len], res$forecast[1:fc_len])
            res$smape <- smape(y_test[1:fc_len], res$forecast[1:fc_len])
            res$mase <- mase(y_test[1:fc_len], res$forecast[1:fc_len], y_train, seasonality)
        }
        results$smooth_opt <- res
    }

    # forecast::ets
    if ("ets" %in% methods) {
        res <- benchmark_forecast_ets(data)
        if (res$success) {
            fc_len <- min(length(res$forecast), length(y_test))
            res$mae <- mae(y_test[1:fc_len], res$forecast[1:fc_len])
            res$rmse <- rmse(y_test[1:fc_len], res$forecast[1:fc_len])
            res$smape <- smape(y_test[1:fc_len], res$forecast[1:fc_len])
            res$mase <- mase(y_test[1:fc_len], res$forecast[1:fc_len], y_train, seasonality)
        }
        results$ets <- res
    }

    return(results)
}

# =============================================================================
# Main Benchmark Runner
# =============================================================================

run_benchmark <- function(datasets, dataset_name, n_series = NULL,
                          methods = c("smooth_back", "smooth_opt", "ets")) {
    # Run benchmark on a set of Mcomp-format series

    if (!is.null(n_series) && n_series < length(datasets)) {
        datasets <- datasets[1:n_series]
    }

    cat(sprintf("\n%s\n", paste(rep("=", 60), collapse = "")))
    cat(sprintf("Benchmarking: %s (%d series)\n", dataset_name, length(datasets)))
    cat(sprintf("%s\n", paste(rep("=", 60), collapse = "")))

    all_results <- data.frame()

    pb <- txtProgressBar(min = 0, max = length(datasets), style = 3)

    for (i in seq_along(datasets)) {
        setTxtProgressBar(pb, i)

        data <- datasets[[i]]
        series_results <- benchmark_single_series(data, methods)

        for (method_name in names(series_results)) {
            res <- series_results[[method_name]]

            row <- data.frame(
                dataset = dataset_name,
                series_id = i,
                method = method_name,
                time = res$time,
                mae = ifelse(res$success, res$mae, NA),
                rmse = ifelse(res$success, res$rmse, NA),
                smape = ifelse(res$success, res$smape, NA),
                mase = ifelse(res$success, res$mase, NA),
                model_type = res$model_type,
                success = res$success,
                stringsAsFactors = FALSE
            )
            all_results <- rbind(all_results, row)
        }
    }

    close(pb)
    cat("\n")

    return(all_results)
}

aggregate_results <- function(results_df) {
    # Compute summary statistics by method
    successful <- results_df[results_df$success, ]

    summary_df <- aggregate(
        cbind(mae, rmse, smape, mase, time) ~ dataset + method,
        data = successful,
        FUN = mean,
        na.rm = TRUE
    )

    # Add count
    count_df <- aggregate(success ~ dataset + method, data = successful, FUN = sum)
    summary_df$n_series <- count_df$success

    return(summary_df)
}

print_summary <- function(summary_df) {
    cat("\n")
    cat(paste(rep("=", 80), collapse = ""))
    cat("\nBENCHMARK SUMMARY\n")
    cat(paste(rep("=", 80), collapse = ""))
    cat("\n")

    datasets <- unique(summary_df$dataset)

    for (ds in datasets) {
        cat(sprintf("\nDataset: %s\n", ds))
        cat(paste(rep("-", 70), collapse = ""))
        cat("\n")

        cat(sprintf("%-18s %10s %10s %10s %8s %10s\n",
                    "Method", "MAE", "RMSE", "sMAPE", "MASE", "Avg Time"))
        cat(paste(rep("-", 70), collapse = ""))
        cat("\n")

        ds_summary <- summary_df[summary_df$dataset == ds, ]

        for (j in 1:nrow(ds_summary)) {
            row <- ds_summary[j, ]
            cat(sprintf("%-18s %10.2f %10.2f %9.2f%% %8.3f %9.4fs\n",
                        row$method, row$mae, row$rmse, row$smape, row$mase, row$time))
        }

        cat(paste(rep("-", 70), collapse = ""))
        cat("\n")
    }
}

# =============================================================================
# Main Entry Point
# =============================================================================

main <- function() {
    # Parse command line arguments
    args <- commandArgs(trailingOnly = TRUE)

    n_series <- NULL
    run_m3 <- TRUE
    run_m1 <- TRUE
    run_tourism <- FALSE  # Only if Tcomp is available
    subset_filter <- NULL

    for (i in seq_along(args)) {
        if (args[i] == "--n_series" && i < length(args)) {
            n_series <- as.integer(args[i + 1])
        } else if (args[i] == "--dataset" && i < length(args)) {
            dataset_arg <- toupper(args[i + 1])
            run_m3 <- dataset_arg == "M3"
            run_m1 <- dataset_arg == "M1"
            run_tourism <- dataset_arg == "TOURISM"
        } else if (args[i] == "--subset" && i < length(args)) {
            subset_filter <- toupper(args[i + 1])
        }
    }

    all_results <- data.frame()

    # M3 Competition
    if (run_m3) {
        subsets <- c("MONTHLY", "QUARTERLY", "YEARLY", "OTHER")
        if (!is.null(subset_filter)) {
            subsets <- subsets[subsets == subset_filter]
        }

        for (subset_name in subsets) {
            datasets <- subset(M3, subset_name)
            if (length(datasets) > 0) {
                results <- run_benchmark(datasets, paste0("M3_", tolower(subset_name)), n_series)
                all_results <- rbind(all_results, results)
            }
        }
    }

    # M1 Competition
    if (run_m1) {
        subsets <- c("MONTHLY", "YEARLY")
        if (!is.null(subset_filter)) {
            subsets <- subsets[subsets == subset_filter]
        }

        for (subset_name in subsets) {
            datasets <- subset(M1, subset_name)
            if (length(datasets) > 0) {
                results <- run_benchmark(datasets, paste0("M1_", tolower(subset_name)), n_series)
                all_results <- rbind(all_results, results)
            }
        }
    }

    # Tourism Competition (if Tcomp package is available)
    if (run_tourism && has_tcomp) {
        subsets <- c("MONTHLY", "QUARTERLY", "YEARLY")
        if (!is.null(subset_filter)) {
            subsets <- subsets[subsets == subset_filter]
        }

        for (subset_name in subsets) {
            datasets <- subset(tourism, subset_name)
            if (length(datasets) > 0) {
                results <- run_benchmark(datasets, paste0("tourism_", tolower(subset_name)), n_series)
                all_results <- rbind(all_results, results)
            }
        }
    } else if (run_tourism && !has_tcomp) {
        cat("\nWarning: Tcomp package not installed. Skipping Tourism dataset.\n")
        cat("Install with: install.packages('Tcomp')\n")
    }

    # Print and save results
    if (nrow(all_results) > 0) {
        summary_df <- aggregate_results(all_results)
        print_summary(summary_df)

        # Save results
        output_file <- "results_R_benchmark.csv"
        write.csv(all_results, output_file, row.names = FALSE)
        cat(sprintf("\nDetailed results saved to: %s\n", output_file))

        summary_file <- "results_R_summary.csv"
        write.csv(summary_df, summary_file, row.names = FALSE)
        cat(sprintf("Summary saved to: %s\n", summary_file))
    }
}

# Run main if script is executed directly
if (!interactive()) {
    main()
}
