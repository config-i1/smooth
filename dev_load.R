# Development mode loader for smooth package
# Run this script to load the package in development mode

if (!requireNamespace("devtools", quietly = TRUE)) {
  install.packages("devtools", repos = "https://cran.rstudio.com/")
}

# Load the package in development mode
devtools::load_all()

# Print confirmation
cat("smooth package loaded in development mode.\n")
cat("Any changes to R/ files will be reflected when you run this script again.\n") 