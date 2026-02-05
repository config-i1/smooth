import numpy as np


def _expand_orders(orders):
    """
    Expand ARIMA orders into AR, I, MA components.

    Parameters
    ----------
    orders : list, tuple, int, dict, or None
        ARIMA order specification. Can be:
        - dict with 'ar', 'i', 'ma' keys
        - list/tuple of [ar, i, ma] values
        - single int (interpreted as AR order)

    Returns
    -------
    tuple
        (ar_orders, i_orders, ma_orders)
    """
    # Default values
    ar_orders = i_orders = ma_orders = [0]

    if orders is None:
        return ar_orders, i_orders, ma_orders

    # Handle dict input (from ADAM class)
    if isinstance(orders, dict):
        ar = orders.get("ar", 0)
        i = orders.get("i", 0)
        ma = orders.get("ma", 0)
        ar_orders = [ar] if isinstance(ar, (int, float)) else list(ar) if ar else [0]
        i_orders = [i] if isinstance(i, (int, float)) else list(i) if i else [0]
        ma_orders = [ma] if isinstance(ma, (int, float)) else list(ma) if ma else [0]
    # Handle list/tuple input
    elif isinstance(orders, (list, tuple)):
        if len(orders) >= 3:
            ar_orders = (
                [orders[0]] if isinstance(orders[0], (int, float)) else orders[0]
            )
            i_orders = [orders[1]] if isinstance(orders[1], (int, float)) else orders[1]
            ma_orders = (
                [orders[2]] if isinstance(orders[2], (int, float)) else orders[2]
            )
    elif isinstance(orders, (int, float)):
        # Single value -> assume AR order
        ar_orders = [orders]

    return ar_orders, i_orders, ma_orders


def _check_arima(orders, validated_lags, silent=False):
    """
    Check and validate ARIMA model specification.

    This function mirrors R's parametersChecker ARIMA handling in adamGeneral.R lines
    519-666.

    Parameters
    ----------
    orders : list, tuple, int, or None
        ARIMA order specification
    validated_lags : list
        List of validated lags (must include 1 as first element for non-seasonal)
    silent : bool, optional
        Whether to suppress warnings

    Returns
    -------
    dict
        Dictionary with ARIMA model information including:
        - non_zero_ari: Nx2 matrix [polynomial_index, state_index] (0-indexed for
        Python)
        - non_zero_ma: Nx2 matrix [polynomial_index, state_index] (0-indexed for Python)
        - lags_model_arima: list of lag values for ARIMA states
        - components_number_arima: number of ARIMA state components
    """

    # Initialize with default values - R lines 652-666
    arima_result = {
        "arima_model": False,
        "ar_orders": None,
        "i_orders": None,
        "ma_orders": None,
        "ar_required": False,
        "i_required": False,
        "ma_required": False,
        "ar_estimate": False,
        "ma_estimate": False,
        "arma_parameters": None,
        "lags_model_arima": [],
        "non_zero_ari": np.zeros((0, 2), dtype=int),
        "non_zero_ma": np.zeros((0, 2), dtype=int),
        "components_number_arima": 0,
        "components_names_arima": [],
        "initial_arima_number": 0,
        "select": False,
    }

    # If no orders specified, return default values
    if orders is None:
        return arima_result

    # Parse orders into components - R lines 521-538
    ar_orders, i_orders, ma_orders = _expand_orders(orders)

    # Check for valid ARIMA component - R lines 541-542
    if (sum(ar_orders) + sum(i_orders) + sum(ma_orders)) == 0:
        return arima_result

    arima_result["arima_model"] = True

    # See if AR/I/MA is needed - R lines 544-560
    ar_required = sum(ar_orders) > 0
    i_required = sum(i_orders) > 0
    ma_required = sum(ma_orders) > 0

    arima_result["ar_required"] = ar_required
    arima_result["i_required"] = i_required
    arima_result["ma_required"] = ma_required

    # Ensure lags start with 1
    lags = list(validated_lags) if validated_lags else [1]
    if lags[0] != 1:
        lags = [1] + lags

    # Define maxOrder and align orders with lags - R lines 563-578
    max_order = max(len(ar_orders), len(i_orders), len(ma_orders), len(lags))

    # Pad orders with zeros to match max_order
    ar_orders = list(ar_orders) + [0] * (max_order - len(ar_orders))
    i_orders = list(i_orders) + [0] * (max_order - len(i_orders))
    ma_orders = list(ma_orders) + [0] * (max_order - len(ma_orders))

    # If lags shorter than max_order, filter orders by non-zero lags - R lines 573-578
    if len(lags) < max_order:
        lags_new = list(lags) + [0] * (max_order - len(lags))
        # Filter to keep only orders where lags are non-zero
        ar_orders = [ar_orders[i] for i in range(len(lags_new)) if lags_new[i] != 0]
        i_orders = [i_orders[i] for i in range(len(lags_new)) if lags_new[i] != 0]
        ma_orders = [ma_orders[i] for i in range(len(lags_new)) if lags_new[i] != 0]
    else:
        # Make sure lags matches the length
        lags = list(lags[:max_order])

    # If after filtering all orders are zero, return no ARIMA - R lines 632-646
    if all(o == 0 for o in ar_orders + i_orders + ma_orders):
        return arima_result

    arima_result["ar_orders"] = ar_orders
    arima_result["i_orders"] = i_orders
    arima_result["ma_orders"] = ma_orders

    # Define the non-zero values via polynomial computation - R lines 580-616
    # This computes all possible lag positions in the ARI and MA polynomials
    ari_values = []
    ma_values = []

    for i in range(len(lags)):
        # ARI values for this lag - R lines 583-588
        ari_for_lag = [0]
        if ar_orders[i] > 0:
            ari_for_lag.extend(range(1, ar_orders[i] + 1))
        if i_orders[i] > 0:
            ari_for_lag.extend(range(ar_orders[i] + 1, ar_orders[i] + i_orders[i] + 1))
        # Multiply by lag and take unique - R line 588
        ari_for_lag = list(set([v * lags[i] for v in ari_for_lag]))
        ari_values.append(ari_for_lag)

        # MA values for this lag - R line 589
        ma_for_lag = [0]
        if ma_orders[i] > 0:
            ma_for_lag.extend(range(1, ma_orders[i] + 1))
        ma_for_lag = list(set([v * lags[i] for v in ma_for_lag]))
        ma_values.append(ma_for_lag)

    # Produce ARI polynomial lag positions - R lines 592-603
    # This creates all combinations of lag positions across seasonal factors
    def expand_polynomial(values_list):
        """Expand polynomial by multiplying across seasonal factors."""
        if len(values_list) == 0:
            return [0]
        result = values_list[0]
        for i in range(1, len(values_list)):
            new_result = []
            for r in result:
                for v in values_list[i]:
                    new_result.append(r + v)
            result = new_result
        return result

    ari_polynomial = expand_polynomial(ari_values)
    ma_polynomial = expand_polynomial(ma_values)

    # What are the non-zero ARI and MA polynomials? - R lines 618-625
    # Remove the first element (which corresponds to L^0 = 1 coefficient) and get unique
    non_zero_ari_lags = sorted(set([x for x in ari_polynomial if x > 0]))
    non_zero_ma_lags = sorted(set([x for x in ma_polynomial if x > 0]))

    # Lags for the ARIMA components - R line 623
    lags_model_arima = sorted(set(non_zero_ari_lags + non_zero_ma_lags))

    if len(lags_model_arima) == 0:
        # No ARIMA states needed
        arima_result["arima_model"] = False
        return arima_result

    # Create nonZeroARI matrix - R line 624
    # Column 0: polynomial index (position in ariPolynomial, 0-indexed for Python)
    # Column 1: state index (position in lagsModelARIMA, 0-indexed for Python)
    non_zero_ari = []
    for lag in non_zero_ari_lags:
        poly_idx = lag  # The lag value itself serves as index into polynomial
        state_idx = lags_model_arima.index(lag)
        non_zero_ari.append([poly_idx, state_idx])

    non_zero_ma = []
    for lag in non_zero_ma_lags:
        poly_idx = lag
        state_idx = lags_model_arima.index(lag)
        non_zero_ma.append([poly_idx, state_idx])

    # Convert to numpy arrays
    non_zero_ari = (
        np.array(non_zero_ari, dtype=int)
        if non_zero_ari
        else np.zeros((0, 2), dtype=int)
    )
    non_zero_ma = (
        np.array(non_zero_ma, dtype=int) if non_zero_ma else np.zeros((0, 2), dtype=int)
    )

    # Number of components - R line 628
    components_number_arima = len(lags_model_arima)

    # Component names - R lines 629-630
    if components_number_arima > 1:
        components_names_arima = [
            f"ARIMAState{i + 1}" for i in range(components_number_arima)
        ]
    else:
        components_names_arima = ["ARIMAState1"] if components_number_arima > 0 else []

    # Number of initials needed - R line 649
    initial_arima_number = max(lags_model_arima) if lags_model_arima else 0

    # Update result
    arima_result["non_zero_ari"] = non_zero_ari
    arima_result["non_zero_ma"] = non_zero_ma
    arima_result["lags_model_arima"] = lags_model_arima
    arima_result["components_number_arima"] = components_number_arima
    arima_result["components_names_arima"] = components_names_arima
    arima_result["initial_arima_number"] = initial_arima_number

    # Set estimation flags - always estimate parameters if they are required
    arima_result["ar_estimate"] = ar_required
    arima_result["ma_estimate"] = ma_required

    # Initialize ARMA parameters (will be filled during estimation)
    arima_parameters = []
    if ar_required:
        for i in range(len(ar_orders)):
            for j in range(ar_orders[i]):
                arima_parameters.append(0.0)
    if ma_required:
        for i in range(len(ma_orders)):
            for j in range(ma_orders[i]):
                arima_parameters.append(0.0)

    arima_result["arma_parameters"] = arima_parameters if arima_parameters else None

    return arima_result
