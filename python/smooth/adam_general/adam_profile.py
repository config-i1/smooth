import numpy as np
import pandas as pd
from typing import Union, List, Dict, Any

def adam_profile_creator(
    lags_model_all: List[List[int]],
    lags_model_max: int,
    obs_all: int,
    lags: Union[List[int], None] = None,
    y_index: Union[List, None] = None,
    y_classes: Union[List, None] = None
) -> Dict[str, np.ndarray]:
    """
    Creates recent profile and the lookup table for ADAM.

    Args:
        lags_model_all: All lags used in the model for ETS + ARIMA + xreg.
        lags_model_max: The maximum lag used in the model.
        obs_all: Number of observations to create.
        lags: The original lags provided by user (optional).
        y_index: The indices needed to get the specific dates (optional).
        y_classes: The class used for the actual data (optional).

    Returns:
        A dictionary with 'recent' (profiles_recent_table) and 'lookup'
        (index_lookup_table) as keys.
    """
    # Initialize matrices
    profiles_recent_table = np.zeros((len(lags_model_all), lags_model_max))
    index_lookup_table = np.ones((len(lags_model_all), obs_all + lags_model_max))
    profile_indices = (
        np.arange(1, lags_model_max * len(lags_model_all) + 1)
        .reshape(-1, len(lags_model_all))
        .T
    )

    # Update matrices based on lagsModelAll
    for i, lag in enumerate(lags_model_all):
        # Create the matrix with profiles based on the provided lags.
        # For every row, fill the first 'lag' elements from 1 to lag
        profiles_recent_table[i, : lag[0]] = np.arange(1, lag[0] + 1)

        # For the i-th row in indexLookupTable, fill with a repeated sequence starting
        # from lagsModelMax to the end of the row.
        # The repeated sequence is the i-th row of profileIndices, repeated enough times
        # to cover 'obsAll' observations.
        # '- 1' at the end adjusts these values to Python's zero-based indexing.
        index_lookup_table[i, lags_model_max : (lags_model_max + obs_all)] = (
            np.tile(
                profile_indices[i, : lags_model_all[i][0]],
                int(np.ceil(obs_all / lags_model_all[i][0])),
            )[0:obs_all]
            - 1
        )

        # Extract unique values from from lagsModelMax to lagsModelMax + obsAll of
        # indexLookupTable
        unique_values = np.unique(
            index_lookup_table[i, lags_model_max : lags_model_max + obs_all]  # noqa
        )

        # fix the head of teh data before the sample starts
        # Repeat the unique values lagsModelMax times and then trim the sequence to only
        # keep the first lagsModelMax elements
        index_lookup_table[i, :lags_model_max] = np.tile(unique_values, lags_model_max)[
            -lags_model_max:
        ]

    # Convert to int!
    index_lookup_table = index_lookup_table.astype(int)

    # Note: I skip andling of special cases (e.g., daylight saving time, leap years)
    return {
        "recent": np.array(profiles_recent_table, dtype="float64"),
        "lookup": np.array(index_lookup_table, dtype="int64"),
    }

def parameters_checker(data: Union[np.ndarray, pd.Series, pd.DataFrame],
                       model: str = "ZXZ",
                       lags: Union[List[int], None] = None,
                       orders: Union[List[int], Dict[str, List[int]], None] = None,
                       formula: Union[str, None] = None,
                       constant: bool = True,
                       distribution: str = "dnorm",
                       loss: str = "likelihood",
                       h: int = 10,
                       holdout: bool = False,
                       persistence: Union[List[float], Dict[str, Union[float, List[float]]], None] = None,
                       phi: Union[float, None] = None,
                       initial: Union[str, List[float], Dict[str, Union[float, List[float]]], None] = None,
                       **kwargs: Any) -> Dict[str, Any]:
    """
    Checks and processes input parameters for ADAM models.

    Args:
        data: Input data (numpy array, pandas Series, or DataFrame).
        model: Model type string.
        lags: List of lag values.
        orders: ARIMA orders as list [p, d, q] or dict {'ar': p, 'i': d, 'ma': q}.
        formula: Formula string for regression component.
        constant: Whether to include a constant term.
        distribution: Error distribution type.
        loss: Loss function to use.
        h: Forecast horizon.
        holdout: Whether to use holdout for evaluation.
        persistence: Persistence parameters.
        phi: Damping parameter.
        initial: Initial values for model components.
        **kwargs: Additional keyword arguments.

    Returns:
        A dictionary of processed parameters.

    Raises:
        ValueError: If any of the input parameters are invalid.
    """
    # Check data
    if isinstance(data, pd.DataFrame):
        y = data.iloc[:, 0].values
        xreg = data.iloc[:, 1:].values if data.shape[1] > 1 else None
    elif isinstance(data, pd.Series):
        y = data.values
        xreg = None
    elif isinstance(data, np.ndarray):
        if data.ndim == 1:
            y = data
            xreg = None
        elif data.ndim == 2:
            y = data[:, 0]
            xreg = data[:, 1:] if data.shape[1] > 1 else None
        else:
            raise ValueError("data must be 1D or 2D array-like")
    else:
        raise ValueError("data must be pandas DataFrame, Series, or numpy array")

    # Check model
    if not isinstance(model, str):
        raise ValueError("model must be a string")
    
    # Check lags
    if lags is None:
        lags = [1]  # Default to 1 if not provided
    if not isinstance(lags, list):
        raise ValueError("lags must be a list of integers or None")
    
    # Check orders
    if orders is not None:
        if isinstance(orders, list):
            if len(orders) != 3:
                raise ValueError("orders as list must have 3 elements: [p, d, q]")
        elif isinstance(orders, dict):
            if not all(key in orders for key in ['ar', 'i', 'ma']):
                raise ValueError("orders as dict must have keys: 'ar', 'i', 'ma'")
        else:
            raise ValueError("orders must be a list, dict, or None")
    
    # Check formula
    if formula is not None and not isinstance(formula, str):
        raise ValueError("formula must be a string or None")
    
    # Check distribution
    valid_distributions = ["dnorm", "dlaplace", "ds", "dgnorm", "dlnorm", "dgamma", "dinvgauss"]
    if distribution not in valid_distributions:
        raise ValueError(f"distribution must be one of {valid_distributions}")
    
    # Check loss
    valid_losses = ["likelihood", "MSE", "MAE", "HAM", "LASSO", "RIDGE", "TMSE", "GTMSE", "MSEh", "MSCE"]
    if loss not in valid_losses and not callable(loss):
        raise ValueError(f"loss must be one of {valid_losses} or a callable function")
    
    # Check h and holdout
    if not isinstance(h, int) or h <= 0:
        raise ValueError("h must be a positive integer")
    if not isinstance(holdout, bool):
        raise ValueError("holdout must be a boolean")
    
    # Check persistence
    if persistence is not None:
        if not isinstance(persistence, (list, dict)):
            raise ValueError("persistence must be a list, dict, or None")
    
    # Check phi
    if phi is not None and not isinstance(phi, (int, float)):
        raise ValueError("phi must be a number or None")
    
    # Check initial
    valid_initial_str = ["optimal", "backcasting", "complete"]
    if initial is not None:
        if isinstance(initial, str) and initial not in valid_initial_str:
            raise ValueError(f"initial as string must be one of {valid_initial_str}")
        elif not isinstance(initial, (str, list, dict)):
            raise ValueError("initial must be a string, list, dict, or None")
    
    # Return the processed parameters
    return {
        "y": y,
        "xreg": xreg,
        "model": model,
        "lags": lags,
        "orders": orders,
        "formula": formula,
        "constant": constant,
        "distribution": distribution,
        "loss": loss,
        "h": h,
        "holdout": holdout,
        "persistence": persistence,
        "phi": phi,
        "initial": initial,
        **kwargs
    }

def architector(ets_model: bool, E_type: str, T_type: str, S_type: str,
                lags: List[int], lags_model_seasonal: List[int],
                xreg_number: int, obs_in_sample: int, initial_type: str,
                arima_model: bool, lags_model_ARIMA: List[int],
                xreg_model: bool, constant_required: bool,
                profiles_recent_table: Union[np.ndarray, None] = None,
                profiles_recent_provided: bool = False) -> Dict[str, Any]:
    """
    Constructs the architecture for ADAM models.

    Args:
        ets_model: Whether ETS model is included.
        E_type, T_type, S_type: ETS model types for error, trend, and seasonality.
        lags: List of lag values.
        lags_model_seasonal: List of seasonal lags.
        xreg_number: Number of external regressors.
        obs_in_sample: Number of in-sample observations.
        initial_type: Type of initial values.
        arima_model: Whether ARIMA model is included.
        lags_model_ARIMA: List of ARIMA lags.
        xreg_model: Whether external regressors are included.
        constant_required: Whether a constant term is required.
        profiles_recent_table: Pre-computed recent profiles table (optional).
        profiles_recent_provided: Whether profiles_recent_table is provided.

    Returns:
        A dictionary containing the model architecture components.
    """
    components = {}

    # If there is ETS
    if ets_model:
        model_is_trendy = T_type != "N"
        if model_is_trendy:
            # Make lags (1, 1)
            lags_model = [[1, 1]]
            components_names_ETS = ["level", "trend"]
        else:
            # Make lags (1, ...)
            lags_model = [[1]]
            components_names_ETS = ["level"]
        
        model_is_seasonal = S_type != "N"
        if model_is_seasonal:
            # If the lags are for the non-seasonal model
            lags_model.extend([[lag] for lag in lags_model_seasonal])
            components_number_ETS_seasonal = len(lags_model_seasonal)
            if components_number_ETS_seasonal > 1:
                components_names_ETS.extend([f"seasonal{i+1}" for i in range(components_number_ETS_seasonal)])
            else:
                components_names_ETS.append("seasonal")
        else:
            components_number_ETS_seasonal = 0
        
        lags_model_all = lags_model
        components_number_ETS = len(lags_model)
    else:
        model_is_trendy = model_is_seasonal = False
        components_number_ETS = components_number_ETS_seasonal = 0
        components_names_ETS = None
        lags_model_all = lags_model = None

    # If there is ARIMA
    if arima_model:
        lags_model_all = lags_model + [[lag] for lag in lags_model_ARIMA]

    # If constant is needed, add it
    if constant_required:
        lags_model_all.append([1])

    # If there are xreg
    if xreg_model:
        lags_model_all.extend([[1]] * xreg_number)

    lags_model_max = max(max(lag) for lag in lags_model_all)

    # Define the number of cols that should be in the matvt
    obs_states = obs_in_sample + lags_model_max

    # Create ADAM profiles for correct treatment of seasonality
    adam_profiles = adam_profile_creator(lags_model_all, lags_model_max, obs_in_sample + lags_model_max,
                                         lags=lags, y_index=None, y_classes=None)
    if profiles_recent_provided:
        profiles_recent_table = profiles_recent_table[:, :lags_model_max]
    else:
        profiles_recent_table = adam_profiles['recent']
    index_lookup_table = adam_profiles['lookup']

    components.update({
        'model_is_trendy': model_is_trendy,
        'model_is_seasonal': model_is_seasonal,
        'components_number_ETS': components_number_ETS,
        'components_number_ETS_seasonal': components_number_ETS_seasonal,
        'components_names_ETS': components_names_ETS,
        'lags_model': lags_model,
        'lags_model_all': lags_model_all,
        'lags_model_max': lags_model_max,
        'obs_states': obs_states,
        'profiles_recent_table': profiles_recent_table,
        'index_lookup_table': index_lookup_table
    })

    return components