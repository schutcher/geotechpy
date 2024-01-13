import pandas as pd

def split_layer_at_elevation(dataframe, elevation):
    """
    Splits a soil layer in the DataFrame `dataframe` at the specified `elevation`.
    The function will create two new layers with the same properties as the original,
    except for the layer_id and elevations.

    :param dataframe: Pandas DataFrame containing the soil profile data.
    :param elevation: Elevation at which the layer should be split.
    :return: DataFrame with the updated soil profile.
    """
    # Check if the elevation is within the range of the profile
    if (
        elevation > dataframe["top_elevation"].max()
        or elevation < dataframe["bottom_elevation"].min()
    ):
        # print("Water level is outside the soil profile range.")
        return dataframe

    # Find the layer that contains the water level
    layer_to_split = dataframe[
        (dataframe["top_elevation"] > elevation)
        & (dataframe["bottom_elevation"] < elevation)
    ]

    # If no layer is found, the elevation is exactly at a layer boundary
    if layer_to_split.empty:
        # print("Elevation is exactly at a layer boundary.")
        return dataframe

    # Create two new layers from the original layer
    top_layer = layer_to_split.copy()
    bottom_layer = layer_to_split.copy()

    top_layer["bottom_elevation"] = elevation
    bottom_layer["top_elevation"] = elevation

    # Update layer IDs
    original_id = layer_to_split.iloc[0]["layer_id"]
    top_layer["layer_id"] = f"{original_id}a"
    bottom_layer["layer_id"] = f"{original_id}b"

    # Remove the original layer and add the new layers
    dataframe = dataframe[dataframe["layer_id"] != original_id]
    dataframe = pd.concat([dataframe, top_layer, bottom_layer], ignore_index=True)

    # Sort by top elevation to maintain order
    dataframe.sort_values(by="top_elevation", ascending=False, inplace=True)
    dataframe.reset_index(drop=True, inplace=True)

    return dataframe