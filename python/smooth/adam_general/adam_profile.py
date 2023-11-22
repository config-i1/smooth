import numpy as np


def adamProfileCreator(
    lagsModelAll, lagsModelMax, obsAll, lags=None, yIndex=None, yClasses=None
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
    dict: A dictionary with 'recent' (profilesRecentTable) and 'lookup' (indexLookupTable) as keys.
    """
    # Initialize matrices
    profilesRecentTable = np.zeros((len(lagsModelAll), lagsModelMax))
    indexLookupTable = np.ones((len(lagsModelAll), obsAll + lagsModelMax))
    profileIndices = (
        np.arange(1, lagsModelMax * len(lagsModelAll) + 1)
        .reshape(-1, len(lagsModelAll))
        .T
    )

    # Update matrices based on lagsModelAll
    for i, lag in enumerate(lagsModelAll):
        # Create the matrix with profiles based on the provided lags.
        # For every row, fill the first 'lag' elements from 1 to lag
        profilesRecentTable[i, : lag[0]] = np.arange(1, lag[0] + 1)

        # For the i-th row in indexLookupTable, fill with a repeated sequence starting from lagsModelMax to the end of the row.
        # The repeated sequence is the i-th row of profileIndices, repeated enough times to cover 'obsAll' observations.
        # '- 1' at the end adjusts these values to Python's zero-based indexing.
        indexLookupTable[i, lagsModelMax : (lagsModelMax + obsAll)] = (  # noqa
            np.tile(
                profileIndices[i, : lagsModelAll[i][0]],
                int(np.ceil(obsAll / lagsModelAll[i][0])),
            )[0:obsAll]
            - 1
        )

        # Extract unique values from from lagsModelMax to lagsModelMax + obsAll of indexLookupTable
        unique_values = np.unique(
            indexLookupTable[i, lagsModelMax : lagsModelMax + obsAll]  # noqa
        )

        # fix the head of teh data before the sample starts
        # Repeat the unique values lagsModelMax times and then trim the sequence to only keep the first lagsModelMax elements
        indexLookupTable[i, :lagsModelMax] = np.tile(unique_values, lagsModelMax)[
            -lagsModelMax:
        ]

    # Convert to int!
    indexLookupTable = indexLookupTable.astype(int)

    # Note: I skip andling of special cases (e.g., daylight saving time, leap years)
    return {
        "recent": np.array(profilesRecentTable, dtype="float64"),
        "lookup": np.array(indexLookupTable, dtype="int64"),
    }
