def _warn(msg, silent=False):
    """
    Helper to show warnings in a style closer to R.

    Parameters
    ----------
    msg : str
        Warning message
    silent : bool, optional
        Whether to suppress warnings
    """
    if not silent:
        print(f"Warning: {msg}")
