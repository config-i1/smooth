# Extract Competition Datasets from R Packages
#
# This script extracts M1, M3, and Tourism datasets from R packages
# and saves them as CSV files for use in Python benchmarks.
#
# Run once to create the data files:
#   Rscript extract_data_from_R.R
#
# Requirements:
#   install.packages(c("Mcomp", "Tcomp"))

library(Mcomp)

# Try to load Tcomp (may not be installed)
has_tcomp <- suppressWarnings(require(Tcomp, quietly = TRUE))

# Output directory
data_dir <- "./benchmark_data"
dir.create(data_dir, showWarnings = FALSE, recursive = TRUE)

# =============================================================================
# Helper function to extract and save dataset
# =============================================================================

extract_and_save <- function(datasets, dataset_name, output_dir) {
    cat(sprintf("Extracting %s (%d series)...\n", dataset_name, length(datasets)))

    # Create output subdirectory
    out_path <- file.path(output_dir, dataset_name)
    dir.create(out_path, showWarnings = FALSE, recursive = TRUE)

    # Metadata for all series
    metadata <- data.frame(
        series_id = character(),
        n_train = integer(),
        n_test = integer(),
        horizon = integer(),
        frequency = integer(),
        type = character(),
        stringsAsFactors = FALSE
    )

    for (i in seq_along(datasets)) {
        series <- datasets[[i]]
        series_id <- sprintf("series_%04d", i)

        # Extract data
        train_data <- as.vector(series$x)
        test_data <- as.vector(series$xx)
        horizon <- series$h
        freq <- frequency(series$x)
        series_type <- ifelse(is.null(series$type), "unknown", series$type)

        # Save train data
        train_df <- data.frame(
            t = seq_along(train_data),
            y = train_data
        )
        write.csv(train_df, file.path(out_path, paste0(series_id, "_train.csv")),
                  row.names = FALSE)

        # Save test data
        test_df <- data.frame(
            t = seq_along(test_data),
            y = test_data
        )
        write.csv(test_df, file.path(out_path, paste0(series_id, "_test.csv")),
                  row.names = FALSE)

        # Add to metadata
        metadata <- rbind(metadata, data.frame(
            series_id = series_id,
            n_train = length(train_data),
            n_test = length(test_data),
            horizon = horizon,
            frequency = freq,
            type = series_type,
            stringsAsFactors = FALSE
        ))
    }

    # Save metadata
    write.csv(metadata, file.path(out_path, "metadata.csv"), row.names = FALSE)

    cat(sprintf("  Saved to: %s\n", out_path))
    cat(sprintf("  Series: %d, Total files: %d\n", nrow(metadata), nrow(metadata) * 2 + 1))

    return(metadata)
}

# =============================================================================
# Extract M3 Competition Data
# =============================================================================

cat("\n=== M3 Competition ===\n")

m3_subsets <- list(
    "m3_monthly" = subset(M3, "MONTHLY"),
    "m3_quarterly" = subset(M3, "QUARTERLY"),
    "m3_yearly" = subset(M3, "YEARLY"),
    "m3_other" = subset(M3, "OTHER")
)

for (name in names(m3_subsets)) {
    if (length(m3_subsets[[name]]) > 0) {
        extract_and_save(m3_subsets[[name]], name, data_dir)
    }
}

# =============================================================================
# Extract M1 Competition Data
# =============================================================================

cat("\n=== M1 Competition ===\n")

m1_subsets <- list(
    "m1_monthly" = subset(M1, "MONTHLY"),
    "m1_quarterly" = subset(M1, "QUARTERLY"),
    "m1_yearly" = subset(M1, "YEARLY")
)

for (name in names(m1_subsets)) {
    if (length(m1_subsets[[name]]) > 0) {
        extract_and_save(m1_subsets[[name]], name, data_dir)
    }
}

# =============================================================================
# Extract Tourism Competition Data (if Tcomp available)
# =============================================================================

if (has_tcomp) {
    cat("\n=== Tourism Competition ===\n")

    tourism_subsets <- list(
        "tourism_monthly" = subset(tourism, "MONTHLY"),
        "tourism_quarterly" = subset(tourism, "QUARTERLY"),
        "tourism_yearly" = subset(tourism, "YEARLY")
    )

    for (name in names(tourism_subsets)) {
        if (length(tourism_subsets[[name]]) > 0) {
            extract_and_save(tourism_subsets[[name]], name, data_dir)
        }
    }
} else {
    cat("\n=== Tourism Competition ===\n")
    cat("Skipped: Tcomp package not installed.\n")
    cat("Install with: install.packages('Tcomp')\n")
}

# =============================================================================
# Create summary file
# =============================================================================

cat("\n=== Creating Summary ===\n")

# List all extracted datasets
all_datasets <- list.dirs(data_dir, recursive = FALSE, full.names = FALSE)

summary_data <- data.frame()
for (ds in all_datasets) {
    meta_path <- file.path(data_dir, ds, "metadata.csv")
    if (file.exists(meta_path)) {
        meta <- read.csv(meta_path)
        summary_data <- rbind(summary_data, data.frame(
            dataset = ds,
            n_series = nrow(meta),
            horizon = meta$horizon[1],
            frequency = meta$frequency[1],
            stringsAsFactors = FALSE
        ))
    }
}

write.csv(summary_data, file.path(data_dir, "datasets_summary.csv"), row.names = FALSE)

cat("\nExtraction complete!\n")
cat(sprintf("Data saved to: %s\n", normalizePath(data_dir)))
cat("\nDatasets extracted:\n")
print(summary_data)

cat("\nTo use in Python:\n")
cat("  python benchmark_ets.py --data_dir ./benchmark_data --dataset m3_monthly\n")
