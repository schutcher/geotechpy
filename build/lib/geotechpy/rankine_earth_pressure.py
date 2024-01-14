import pandas as pd
from numpy import radians, cos, sqrt
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon



class RankineEarthPressure:
    """
    Class for calculating Rankine earth pressure coefficients and pressures.

    Args:
        dataframe (pandas.DataFrame): The input dataframe containing the necessary data.
        backfill_slope_angle (float): The angle of the backfill slope in degrees.

    Attributes:
        dataframe (pandas.DataFrame): The input dataframe containing the necessary data.
        backfill_slope_angle (float): The angle of the backfill slope in degrees.

    Methods:
        calculate_active_coefficient: Calculates the active coefficient.
        calculate_passive_coefficient: Calculates the passive coefficient.
        calculate_top_active_pressure: Calculates the top active pressure.
        calculate_bottom_active_pressure: Calculates the bottom active pressure.
        calculate_top_passive_pressure: Calculates the top passive pressure.
        calculate_bottom_passive_pressure: Calculates the bottom passive pressure.
        calculate_coefficients: Calculates both active and passive coefficients.
        calculate_all: Calculates all the coefficients and pressures.
        plot_active_pressures: Plots the soil layers with active pressure polygons.
        plot_passive_pressures: Plots the soil layers with passive pressure polygons.
    """
    def __init__(self, dataframe, backfill_slope_angle):
        self.dataframe = dataframe
        self.backfill_slope_angle = backfill_slope_angle

    @property
    def dataframe(self):
        return self._dataframe
    
    @dataframe.setter
    def dataframe(self, df):
        if not isinstance(df, pd.DataFrame):
            raise TypeError("dataframe must be a pandas DataFrame")
        self._dataframe = df

    @property
    def backfill_slope_angle(self):
        return self._backfill_slope_angle

    @backfill_slope_angle.setter
    def backfill_slope_angle(self, value):
        if not isinstance(value, (int, float)):
            raise TypeError("backfill_slope_angle must be a number")
        self._backfill_slope_angle = value


    def calculate_active_coefficient(self):
        """
        Calculates the active coefficient of lateral earth pressure and updates 
        the 'coulomb_active_coefficient' column in the dataframe attribute of this instance.

        Returns:
            pandas.DataFrame: The updated dataframe with the active coefficient column.
        """
        # calculate active coefficient
        drained_friction_angle_rad = radians(
            self.dataframe["drained_friction_angle"]
        )
        backfill_angle_rad = radians(self.backfill_slope_angle)
        numerator = cos(backfill_angle_rad) - sqrt(
            (cos(backfill_angle_rad)) ** 2
            - (cos(drained_friction_angle_rad)) ** 2
        )
        denominator = cos(backfill_angle_rad) + sqrt(
            (cos(backfill_angle_rad)) ** 2
            - (cos(drained_friction_angle_rad)) ** 2
        )
        active_coefficient = cos(backfill_angle_rad) * numerator / denominator
        self.dataframe["rankine_active_coefficient"] = active_coefficient
        return self.dataframe

    def calculate_passive_coefficient(self):
        """
        Calculates the passive coefficient of lateral earth pressure.

        Returns:
            pandas.DataFrame: The updated dataframe with the passive coefficient column.
        """
        # calculate passive coefficient
        drained_friction_angle_rad = radians(
            self.dataframe["drained_friction_angle"]
        )
        backfill_angle_rad = radians(self.backfill_slope_angle)
        numerator = cos(backfill_angle_rad) + sqrt(
            (cos(backfill_angle_rad)) ** 2
            - (cos(drained_friction_angle_rad)) ** 2
        )
        denominator = cos(backfill_angle_rad) - sqrt(
            (cos(backfill_angle_rad)) ** 2
            - (cos(drained_friction_angle_rad)) ** 2
        )
        passive_coefficient = cos(backfill_angle_rad) * numerator / denominator
        self.dataframe["rankine_passive_coefficient"] = passive_coefficient
        return self.dataframe

    def calculate_top_active_pressure(self):
        """
        Calculates the top active pressure.

        """

        top_active_pressure = round(
            self.dataframe["rankine_active_coefficient"]
            * self.dataframe["top_effective_stress"],
            1,
        )

        self.dataframe["rankine_top_active_pressure"] = top_active_pressure

        return self.dataframe


    def calculate_bottom_active_pressure(self):
        """
        Calculates the bottom active pressure.

        """

        bottom_active_pressure = round(
            self.dataframe["rankine_active_coefficient"]
            * self.dataframe["bottom_effective_stress"],
            1,
        )

        self.dataframe["rankine_bottom_active_pressure"] = bottom_active_pressure

        return self.dataframe


    def calculate_top_passive_pressure(self):
        """
        Calculates the top passive pressure.

        """

        top_passive_pressure = round(
            self.dataframe["rankine_passive_coefficient"]
            * self.dataframe["top_effective_stress"],
            1,
        )

        self.dataframe["rankine_top_passive_pressure"] = top_passive_pressure

        return self.dataframe


    def calculate_bottom_passive_pressure(self):
        """
        Calculates the bottom passive pressure.

        """

        bottom_passive_pressure = round(
            self.dataframe["rankine_passive_coefficient"]
            * self.dataframe["bottom_effective_stress"],
            1,
        )

        self.dataframe["rankine_bottom_passive_pressure"] = bottom_passive_pressure

        return self.dataframe


    def calculate_coefficients(self):
        """
        Calculates both active and passive coefficients.

        Returns:
            pandas.DataFrame: The updated dataframe with the active and passive coefficient columns.
        """
        dataframe = self.calculate_active_coefficient()
        dataframe = self.calculate_passive_coefficient()
        return dataframe

    def calculate_all(self):
        """
        Calculates all the coefficients and pressures.

        Returns:
            pandas.DataFrame: The updated dataframe with all the calculated columns.
        """
        dataframe = self.calculate_coefficients()
        dataframe = self.calculate_top_active_pressure()
        dataframe = self.calculate_bottom_active_pressure()
        dataframe = self.calculate_top_passive_pressure()
        dataframe = self.calculate_bottom_passive_pressure()
        return dataframe

    def plot_active_pressures(self):
        """
        Plots the soil layers with filled polygons representing the active pressure.

        """

        # Get the soil layers dataframe
        soil_layers = self.dataframe

        # Calculate maximum pressure for x-axis limit
        max_pressure = max(
            soil_layers["rankine_top_active_pressure"].max(),
            soil_layers["rankine_bottom_active_pressure"].max(),
        )

        # Calculate elevation range for y-axis limit
        max_elevation = soil_layers["top_elevation"].max()
        min_elevation = soil_layers["bottom_elevation"].min()

        # Create the plot
        fig, ax = plt.subplots(figsize=(10, 6))

        # Add soil layers as filled polygons with semi-transparency
        for i, layer in soil_layers.iterrows():
            # Define the coordinates for each corner of the polygon
            left_top = (0, layer["top_elevation"])
            left_bottom = (
                0,
                layer["bottom_elevation"],
            )
            right_bottom = (
                layer["rankine_bottom_active_pressure"],
                layer["bottom_elevation"],
            )
            right_top = (layer["rankine_top_active_pressure"], layer["top_elevation"])

            ka_value = layer["rankine_active_coefficient"]
            name = layer["layer_id"]
            label = f"Layer {name}, Ka = {ka_value:.3f}"  # Updated label

            # Create the polygon and add it to the plot
            polygon = Polygon(
                [left_top, left_bottom, right_bottom, right_top],
                closed=True,
                edgecolor="black",
                facecolor=f"C{i}",
                alpha=0.5,
                label=label,
            )
            ax.add_patch(polygon)

            # Add title
            plt.title("Rankine Active Earth Pressures")

        # Set dynamic limits
        ax.set_xlim(0, max_pressure * 1.1)  # 10% more than max for better visibility
        ax.set_ylim(min_elevation, max_elevation)

        # Add labels, legend, and grid
        ax.set_xlabel("Pressure (psf)")
        ax.set_ylabel("Elevation (ft)")
        ax.legend(title="Soil Layers")
        ax.grid(True)

        # Show the plot
        plt.show()

    def plot_passive_pressures(self):
        """
        Plots the soil layers with filled polygons representing the passive pressure.

        """

        # Get the soil layers dataframe
        soil_layers = self.dataframe

        # Calculate maximum pressure for x-axis limit
        max_pressure = max(
            soil_layers["rankine_top_passive_pressure"].max(),
            soil_layers["rankine_bottom_passive_pressure"].max(),
        )

        # Calculate elevation range for y-axis limit
        max_elevation = soil_layers["top_elevation"].max()
        min_elevation = soil_layers["bottom_elevation"].min()

        # Create the plot
        fig, ax = plt.subplots(figsize=(10, 6))

        # Add soil layers as filled polygons with semi-transparency
        for i, layer in soil_layers.iterrows():
            # Define the coordinates for each corner of the polygon
            left_top = (0, layer["top_elevation"])
            left_bottom = (
                0,
                layer["bottom_elevation"],
            )
            right_bottom = (
                layer["rankine_bottom_passive_pressure"],
                layer["bottom_elevation"],
            )
            right_top = (layer["rankine_top_passive_pressure"], layer["top_elevation"])

            ka_value = layer["rankine_passive_coefficient"]
            name = layer["layer_id"]
            label = f"Layer {name}, Kp = {ka_value:.3f}"  # Updated label

            # Create the polygon and add it to the plot
            polygon = Polygon(
                [left_top, left_bottom, right_bottom, right_top],
                closed=True,
                edgecolor="black",
                facecolor=f"C{i}",
                alpha=0.5,
                label=label,
            )
            ax.add_patch(polygon)

            # Add title
            plt.title("Rankine Passive Earth Pressures")

        # Adjust subplot parameters to fit annotations
        plt.subplots_adjust(right=0.8)

        # Set dynamic limits
        ax.set_xlim(0, max_pressure * 1.1)  # 10% more than max for better visibility
        ax.set_ylim(min_elevation, max_elevation)

        # Add labels, legend, and grid
        ax.set_xlabel("Pressure (psf)")
        ax.set_ylabel("Elevation (ft)")
        ax.legend(title="Soil Layers")
        ax.grid(True)

        # Show the plot
        plt.show()



