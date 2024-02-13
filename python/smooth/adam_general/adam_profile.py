import numpy as np


def adam_profile_creator(
    lags_model_all, lags_model_max, obs_all, lags=None, y_index=None, y_classes=None
):
    """
    Creates recent profile and the lookup table for adam.
    Parameters:
    lagsModelAll (list): All lags used in the model for ETS + ARIMA + xreg.
    lagsModelMax (int): The maximum lag used in the model.
    obsAll (int): Number of observations to create.
    lags (list): The original lags provided by user (optional).
    yIndex (list): The indices needed to get the specific dates (optional).
    yClasses (list): The class used for the actual data (optional).
    Returns:
    dict: A dictionary with 'recent' (profilesRecentTable) and 'lookup'
    (indexLookupTable) as keys.
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
