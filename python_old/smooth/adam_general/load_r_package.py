import os
import rpy2.robjects as robjects

def load_smooth_package():
    """
    Load the smooth R package in development mode.
    This ensures that any changes to the R code are immediately reflected.
    """
    # Get the path to the root of the smooth package
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Navigate up to the root directory (assuming we're in python/smooth/adam_general)
    root_dir = os.path.abspath(os.path.join(current_dir, "../../../"))
    
    # Load devtools and use load_all to load the package in development mode
    r_command = f"""
    if (!requireNamespace("devtools", quietly = TRUE)) {{
      install.packages("devtools", repos = "https://cran.rstudio.com/")
    }}
    devtools::load_all("{root_dir}")
    """
    
    # Execute the R command
    robjects.r(r_command)
    
    print("smooth R package loaded in development mode")
    return True

# If this script is run directly, load the package
if __name__ == "__main__":
    load_smooth_package() 