import pandas as pd
import matplotlib.pyplot as plt


class SoilProfile:
    def __init__(self, dataframe, surcharge_load, water_surface_elev):
        # Initialize SoilProfile with a dataframe, surcharge load, and water surface elevation
        self._dataframe = dataframe
        self._surcharge_load = surcharge_load
        self._water_surface_elev = water_surface_elev

    @property
    def dataframe(self):
        return self._dataframe

    @dataframe.setter
    def dataframe(self, df):
        if not isinstance(df, pd.DataFrame):
            raise ValueError("df must be a pandas DataFrame")
        self._dataframe = df

    @property
    def surcharge_load(self):
        return self._surcharge_load

    @surcharge_load.setter
    def surcharge_load(self, load):
        if not isinstance(load, (int, float)):
            raise ValueError("load must be a number")
        self._surcharge_load = load

    @property
    def water_surface_elev(self):
        return self._water_surface_elev

    @water_surface_elev.setter
    def water_surface_elev(self, elev):
        if not isinstance(elev, (int, float)):
            raise ValueError("elev must be a number")
        self._water_surface_elev = elev

    def split_layer_at_water_level(self):
        """
        Splits a soil layer in the DataFrame `df` at the specified `water_surface_elev`.
        The function will create two new layers with the same properties as the original,
        except for the layer_id and elevations.

        :param df: Pandas DataFrame containing the soil profile data.
        :param water_surface_elev: Elevation at which the layer should be split.
        :return: DataFrame with the updated soil profile.
        """
        # Check if the water level is within the range of the profile
        if (
            self._water_surface_elev > self._dataframe["top_elevation"].max()
            or self._water_surface_elev < self._dataframe["bottom_elevation"].min()
        ):
            # print("Water level is outside the soil profile range.")
            return self._dataframe

        # Find the layer that contains the water level
        layer_to_split = self._dataframe[
            (self._dataframe["top_elevation"] > self._water_surface_elev)
            & (self._dataframe["bottom_elevation"] < self._water_surface_elev)
        ]

        # If no layer is found, the water level is exactly at a layer boundary
        if layer_to_split.empty:
            # print("Water level is exactly at a layer boundary.")
            return self._dataframe

        # Create two new layers from the original layer
        top_layer = layer_to_split.copy()
        bottom_layer = layer_to_split.copy()

        top_layer["bottom_elevation"] = self._water_surface_elev
        bottom_layer["top_elevation"] = self._water_surface_elev

        # Update layer IDs
        original_id = layer_to_split.iloc[0]["layer_id"]
        top_layer["layer_id"] = f"{original_id}a"
        bottom_layer["layer_id"] = f"{original_id}b"

        # Remove the original layer and add the new layers
        self._dataframe = self._dataframe[self._dataframe["layer_id"] != original_id]
        self._dataframe = pd.concat(
            [self._dataframe, top_layer, bottom_layer], ignore_index=True
        )

        # Sort by top elevation to maintain order
        self._dataframe.sort_values(by="top_elevation", ascending=False, inplace=True)
        self._dataframe.reset_index(drop=True, inplace=True)

        return self._dataframe

    def calculate_effective_unit_weight(self):
        """
        Calculate the effective_unit weight for each layer in a soil profile dataframe.

        Creates a new dataframe column to store the moist unit weight for layers
        above the water surface and buoyant unit weight below the water surface.

        Parameters:
        dataframe (DataFrame): soil profile dataframe containing moist_unit_weight,
                    saturated_unit_weight and top_elevation.
        water_surface_elev (float): elevation of water surface
        """
        # Check if dataframe has all required columns
        required_columns = [
            "saturated_unit_weight",
            "moist_unit_weight",
            "top_elevation",
        ]
        for column in required_columns:
            if column not in self._dataframe.columns:
                raise KeyError(f"Missing necessary column: {column}")

        # Remove existing effective_unit_weight column if it exists
        if "effective_unit_weight" in self._dataframe.columns:
            del self._dataframe["effective_unit_weight"]

        # Calculate effective unit weight
        self._dataframe["effective_unit_weight"] = (
            self._dataframe["saturated_unit_weight"] - 62.4
        )
        self._dataframe.loc[
            self._dataframe["top_elevation"] > self._water_surface_elev,
            "effective_unit_weight",
        ] = self._dataframe["moist_unit_weight"]

        return self._dataframe

    def calculate_total_unit_weight(self):
        """
        Calculate the total unit weight for each layer in a soil profile dataframe.

        Creates a new dataframe column to store the moist unit weight for layers
        above the water surface and saturated unit weight below the water surface.

        Parameters:
        dataframe (DataFrame): soil profile dataframe containing moist_unit_weight,
                    saturated_unit_weight and top_elevation.
        water_surface_elev (float): elevation of water surface
        """
        # Check if dataframe has all required columns
        required_columns = [
            "saturated_unit_weight",
            "moist_unit_weight",
            "top_elevation",
        ]
        for column in required_columns:
            if column not in self._dataframe.columns:
                raise KeyError(f"Missing necessary column: {column}")

        # Remove existing total_unit_weight column if it exists
        if "total_unit_weight" in self._dataframe.columns:
            del self._dataframe["total_unit_weight"]

        # Calculate total unit weight
        self._dataframe["total_unit_weight"] = self._dataframe["saturated_unit_weight"]
        self._dataframe.loc[
            self._dataframe["top_elevation"] > self._water_surface_elev,
            "total_unit_weight",
        ] = self._dataframe["moist_unit_weight"]

    def calculate_layer_height(self):
        """
        Calculate the height of each layer in a soil profile dataframe.

        Creates a new dataframe column to store the height of each layer.

        Parameters:
        dataframe (DataFrame): soil profile dataframe containing top_elevation
                    and bottom_elevation.
        """
        # Check if dataframe has all required columns
        required_columns = ["top_elevation", "bottom_elevation"]
        for column in required_columns:
            if column not in self._dataframe.columns:
                raise KeyError(f"Missing necessary column: {column}")

        # Remove existing layer_height column if it exists
        if "layer_height" in self._dataframe.columns:
            del self._dataframe["layer_height"]

        # Calculate layer height
        self._dataframe["layer_height"] = (
            self._dataframe["top_elevation"] - self._dataframe["bottom_elevation"]
        )

        return self._dataframe

    def calculate_effective_stresses(self):
        """
        Calculate the effective vertical stress at the top and bottom of each layer in a soil profile dataframe.

        Creates new dataframe columns to store the top and bottom effective vertical stress for each layer.

        Parameters:
        dataframe (DataFrame): soil profile dataframe containing 'effective_unit_weight' and 'layer_height'.
        surcharge_load (float): surcharge load applied at the top of the soil profile.
        """
        # Check if required columns are in the dataframe
        required_columns = ["effective_unit_weight", "layer_height"]
        for column in required_columns:
            if column not in self._dataframe.columns:
                raise KeyError(f"Missing necessary column: {column}")

        # Remove existing effective stress columns if they exist
        if "top_effective_stress" in self._dataframe.columns:
            del self._dataframe["top_effective_stress"]

        # Initialize the stress at the top of the first layer
        self._dataframe["top_effective_stress"] = 0
        self._dataframe["bottom_effective_stress"] = 0
        self._dataframe.at[0, "top_effective_stress"] = self._surcharge_load

        # Calculate the effective stress at the top and bottom of each layer
        for i in range(len(self._dataframe)):
            if i > 0:
                # The top stress of the current layer is the bottom stress of the previous layer
                self._dataframe["top_effective_stress"] = self._dataframe[
                    "top_effective_stress"
                ].astype(float)
                self._dataframe.at[i, "top_effective_stress"] = self._dataframe.at[
                    i - 1, "bottom_effective_stress"
                ]

            # Calculate the bottom stress of the current layer
            current_layer_stress = (
                self._dataframe.at[i, "effective_unit_weight"]
                * self._dataframe.at[i, "layer_height"]
            )
            self._dataframe["bottom_effective_stress"] = self._dataframe[
                "bottom_effective_stress"
            ].astype(float)
            self._dataframe.at[i, "bottom_effective_stress"] = (
                self._dataframe.at[i, "top_effective_stress"] + current_layer_stress
            )

        return self._dataframe

    def calculate_middle_effective_stress(self):
        """
        Calculate the effective vertical stress at the middle of each layer in a soil profile dataframe.

        Creates a new dataframe column to store the middle effective vertical stress for each layer.

        Parameters:
        dataframe (DataFrame): soil profile dataframe containing 'top_effective_stress' and 'bottom_effective_stress'.

        Returns:
        DataFrame: soil profile dataframe with an additional column 'middle_effective_stress'.
        """
        # Check if required columns are in the dataframe
        required_columns = ["top_effective_stress", "bottom_effective_stress"]
        for column in required_columns:
            if column not in self._dataframe.columns:
                raise KeyError(f"Missing necessary column: {column}")

        # Remove existing middle_effective_stress column if it exists
        if "middle_effective_stress" in self._dataframe.columns:
            del self._dataframe["middle_effective_stress"]

        # Calculate the effective stress at the middle of each layer
        self._dataframe["middle_effective_stress"] = (
            self._dataframe["top_effective_stress"]
            + self._dataframe["bottom_effective_stress"]
        ) / 2

        return self._dataframe

    def calculate_hydrostatic_pressure(self):
        """
        Calculate the hydrostatic water pressure at the top and bottom of each layer in a soil profile dataframe.

        Creates new dataframe columns to store the hydrostatic water pressure at the top and bottom of each layer.

        Parameters:
        dataframe (DataFrame): soil profile dataframe containing 'top_elevation' and 'bottom_elevation'.
        water_surface_elev (float): elevation of the water surface.

        Returns:
        DataFrame: soil profile dataframe with additional columns 'top_water_pressure' and 'bottom_water_pressure'.
        """
        # Check if required columns are in the dataframe
        required_columns = ["top_elevation", "bottom_elevation"]
        for column in required_columns:
            if column not in self._dataframe.columns:
                raise KeyError(f"Missing necessary column: {column}")

        # Remove existing water pressure columns if they exist
        if "top_water_pressure" in self._dataframe.columns:
            del self._dataframe["top_water_pressure"]
        if "bottom_water_pressure" in self._dataframe.columns:
            del self._dataframe["bottom_water_pressure"]

        # Calculate the hydrostatic water pressure at the top and bottom of each layer
        water_unit_weight = 62.4  # pcf
        self._dataframe["top_water_pressure"] = round(
            (self._water_surface_elev - self._dataframe["top_elevation"])
            * water_unit_weight,
            2,
        )
        self._dataframe.loc[
            self._dataframe["top_water_pressure"] < 0, "top_water_pressure"
        ] = 0
        self._dataframe["bottom_water_pressure"] = round(
            (self._water_surface_elev - self._dataframe["bottom_elevation"])
            * water_unit_weight,
            2,
        )
        self._dataframe.loc[
            self._dataframe["bottom_water_pressure"] < 0, "bottom_water_pressure"
        ] = 0

        return self._dataframe

    def calculate_total_stresses(self):
        """
        Calculate the top and bottom total stress for each layer.

        Parameters:
        df (pandas.DataFrame): A dataframe containing the columns 'top_effective_stress', 'bottom_effective_stress',
                            'top_water_pressure', and 'bottom_water_pressure'.

        Returns:
        df (pandas.DataFrame): The input dataframe with two new columns: 'top_total_stress' and 'bottom_total_stress'.

        Raises:
        ValueError: If df is not a pandas DataFrame or if any of the required columns are missing from df.
        """

        # Check if required columns are in the dataframe
        required_columns = [
            "top_effective_stress",
            "bottom_effective_stress",
            "top_water_pressure",
            "bottom_water_pressure",
        ]
        for column in required_columns:
            if column not in self._dataframe.columns:
                raise ValueError(f"Missing necessary column: {column}")

        # Calculate total stress
        self._dataframe["top_total_stress"] = (
            self._dataframe["top_effective_stress"]
            + self._dataframe["top_water_pressure"]
        )
        self._dataframe["bottom_total_stress"] = (
            self._dataframe["bottom_effective_stress"]
            + self._dataframe["bottom_water_pressure"]
        )

        return self._dataframe

    def calculate_middle_total_stress(self):
        """
        Calculate the total stress at the middle of each layer in a soil profile dataframe.

        Creates a new dataframe column to store the total stress at the middle of each layer.

        Parameters:
        dataframe (DataFrame): soil profile dataframe containing 'top_total_stress' and 'bottom_total_stress'.

        Returns:
        DataFrame: soil profile dataframe with an additional column 'middle_total_stress'.
        """
        # Check if required columns are in the dataframe
        required_columns = ["top_total_stress", "bottom_total_stress"]
        for column in required_columns:
            if column not in self._dataframe.columns:
                raise KeyError(f"Missing necessary column: {column}")

        # Calculate the total stress at the middle of each layer
        self._dataframe["middle_total_stress"] = (
            self._dataframe["top_total_stress"] + self._dataframe["bottom_total_stress"]
        ) / 2

        return self._dataframe

    def calculate_all(self):
        """
        Calculate all soil profile functions.

        Runs all soil profile functions in the correct order and creates the following columns:
        'effective_unit_weight', 'total_unit_weight', 'layer_height', 'top_effective_stress', 'bottom_effective_stress',
        'middle_effective_stress', 'top_water_pressure', 'bottom_water_pressure', 'top_total_stress',
        'bottom_total_stress', 'middle_total_stress'.

        Parameters:
        dataframe (DataFrame): soil profile dataframe containing 'top_elevation', 'bottom_elevation',
                    'saturated_unit_weight', 'moist_unit_weight', 'layer_height', 'effective_unit_weight',
                    'top_effective_stress', 'bottom_effective_stress', 'middle_effective_stress',
                    'top_water_pressure', 'bottom_water_pressure', 'top_total_stress', 'bottom_total_stress',
                    'middle_total_stress'.

        Returns:
        DataFrame: soil profile dataframe with all required values for bearing capacity calculation.
        """
        self.split_layer_at_water_level()
        self.calculate_effective_unit_weight()
        self.calculate_total_unit_weight()
        self.calculate_layer_height()
        self.calculate_effective_stresses()
        self.calculate_middle_effective_stress()
        self.calculate_hydrostatic_pressure()
        self.calculate_total_stresses()
        self.calculate_middle_total_stress()

        return self._dataframe

    def plot_profile(self):
        # First, let's create a copy of the dataframe to avoid modifying the original one
        df = self._dataframe.copy()

        # Rename the columns to have a common prefix
        df = df.rename(
            columns={
                "top_elevation": "elevation_top",
                "bottom_elevation": "elevation_bottom",
                "top_effective_stress": "effective_stress_top",
                "bottom_effective_stress": "effective_stress_bottom",
                "top_water_pressure": "water_pressure_top",
                "bottom_water_pressure": "water_pressure_bottom",
                "top_total_stress": "total_stress_top",
                "bottom_total_stress": "total_stress_bottom",
            }
        )

        # Reshape the dataframe
        df = df.melt(
            id_vars=["elevation_top", "elevation_bottom"],
            value_vars=[
                "effective_stress_top",
                "effective_stress_bottom",
                "water_pressure_top",
                "water_pressure_bottom",
                "total_stress_top",
                "total_stress_bottom",
            ],
            var_name="measurement_type",
            value_name="value",
        )

        # Create a new 'elevation' column based on the 'measurement_type' column
        df["elevation"] = df["elevation_top"].where(
            df["measurement_type"].str.contains("top"), df["elevation_bottom"]
        )

        # Create a new 'measurement' column based on the 'measurement_type' column
        df["measurement"] = df["measurement_type"].apply(lambda x: x.split("_")[0])

        # Drop the unnecessary columns
        df = df.drop(columns=["measurement_type", "elevation_top", "elevation_bottom"])

        df = df.sort_values("elevation", ascending=False)

        # Define a dictionary for the colors and line styles
        styles = {
            "effective": {"color": "black", "linestyle": "--"},
            "water": {"color": "blue", "linestyle": "-"},
            "total": {"color": "gray", "linestyle": "-"},
        }

        # Now you can plot 'value' vs 'elevation' for each measurement
        plt.figure(figsize=(6, 10))
        for measurement in [
            "total",
            "water",
            "effective",
        ]:  # Plot in this order to ensure effective is on top
            plt.plot(
                df.loc[df["measurement"] == measurement, "value"],
                df.loc[df["measurement"] == measurement, "elevation"],
                label=measurement,
                color=styles[measurement]["color"],
                linestyle=styles[measurement]["linestyle"],
            )
        plt.xlabel("Value")
        plt.ylabel("Elevation")
        plt.title("Value vs Elevation")
        plt.legend()
        plt.grid(True)
        plt.show()
