import pandas as pd
import numpy as np
from matplotlib.patches import Polygon
import matplotlib.pyplot as plt


class CoulombEarthPressure:
    """
    Class representing Coulomb lateral earth pressure calculations.

    Attributes:
        dataframe (object): Object representing the soil profile.
        wall_top_elevation (float): Elevation of the top of the wall.
        wall_bottom_elevation (float): Elevation of the bottom of the wall.
        backfill_slope_angle (float): Angle of the backfill slope.
        wall_back_face_angle (float): Angle of the wall back face.
        wall_interface_friction_angle (float): Interface friction angle between the wall and the soil.

    Methods:
        calculate_active_coefficient(): Calculates the active coefficient of lateral earth pressure.
        calculate_passive_coefficient(): Calculates the passive coefficient of lateral earth pressure.
        calculate_top_active_pressure(): Calculates the top active pressure.
        calculate_bottom_active_pressure(): Calculates the bottom active pressure.
        calculate_top_passive_pressure(): Calculates the top passive pressure.
        calculate_bottom_passive_pressure(): Calculates the bottom passive pressure.
        calculate_coefficients(): Calculates both active and passive coefficients.
        calculate_all(): Calculates all the parameters including coefficients and pressures.
    """    
    def __init__(
        self,
        dataframe,
        wall_top_elevation,
        wall_bottom_elevation,
        backfill_slope_angle,
        wall_back_face_angle,
        wall_interface_friction_angle,
    ):
        self.dataframe = dataframe
        self.wall_top_elevation = wall_top_elevation
        self.wall_bottom_elevation = wall_bottom_elevation
        self.backfill_slope_angle = backfill_slope_angle
        self.wall_back_face_angle = wall_back_face_angle
        self.wall_interface_friction_angle = wall_interface_friction_angle

    @property
    def dataframe(self):
        return self._dataframe

    @dataframe.setter
    def dataframe(self, df):
        if not isinstance(df, pd.DataFrame):
            raise TypeError("dataframe must be a pandas DataFrame")
        self._dataframe = df

    @property
    def wall_top_elevation(self):
        return self._wall_top_elevation

    @wall_top_elevation.setter
    def wall_top_elevation(self, value):
        if not isinstance(value, (int, float)):
            raise TypeError("wall_top_elevation must be a number")
        self._wall_top_elevation = value

    @property
    def wall_bottom_elevation(self):
        return self._wall_bottom_elevation

    @wall_bottom_elevation.setter
    def wall_bottom_elevation(self, value):
        if not isinstance(value, (int, float)):
            raise TypeError("wall_bottom_elevation must be a number")
        self._wall_bottom_elevation = value

    @property
    def backfill_slope_angle(self):
        return self._backfill_slope_angle

    @backfill_slope_angle.setter
    def backfill_slope_angle(self, value):
        if not isinstance(value, (int, float)):
            raise TypeError("backfill_slope_angle must be a number")
        self._backfill_slope_angle = value

    @property
    def wall_back_face_angle(self):
        return self._wall_back_face_angle

    @wall_back_face_angle.setter
    def wall_back_face_angle(self, value):
        if not isinstance(value, (int, float)):
            raise TypeError("wall_back_face_angle must be a number")
        self._wall_back_face_angle = value

    @property
    def wall_interface_friction_angle(self):
        return self._wall_interface_friction_angle

    @wall_interface_friction_angle.setter
    def wall_interface_friction_angle(self, value):
        if not isinstance(value, (int, float)):
            raise TypeError("wall_interface_friction_angle must be a number")
        self._wall_interface_friction_angle = value

   


    def calculate_active_coefficient(self):
        """
        Calculates the active coefficient of lateral earth pressure and updates 
        the 'coulomb_active_coefficient' column in the dataframe attribute of this instance.

        Note: This method modifies the dataframe in-place.
        """

        drained_friction_angle_rad = np.radians(
            self.dataframe["drained_friction_angle"]
        )
        wall_back_face_angle_rad = np.radians(self.wall_back_face_angle)
        backfill_slope_angle_rad = np.radians(self.backfill_slope_angle)
        wall_interface_friction_angle_rad = np.radians(
            self.wall_interface_friction_angle
        )

        term1 = np.cos(drained_friction_angle_rad - wall_back_face_angle_rad) ** 2
        term2 = np.cos(wall_back_face_angle_rad) ** 2
        term3 = np.cos(wall_interface_friction_angle_rad + wall_back_face_angle_rad)
        term4 = np.sin(wall_interface_friction_angle_rad + drained_friction_angle_rad)
        term5 = np.sin(drained_friction_angle_rad - backfill_slope_angle_rad)
        term6 = term3  # term3 is the same as term6
        term7 = np.cos(wall_back_face_angle_rad - backfill_slope_angle_rad)

        active_coefficient = round(
            term1
            / (term2 * term3 * (1 + np.sqrt((term4 * term5) / (term6 * term7))) ** 2),
            3,
        )

        self.dataframe["coulomb_active_coefficient"] = active_coefficient

        return self.dataframe


    def calculate_passive_coefficient(self):
        """
        Calculates the passive coefficient of lateral earth pressure.

        Returns:
            pandas.DataFrame: The updated dataframe with the passive coefficient column.
        """

        drained_friction_angle_rad = np.radians(
            self.dataframe["drained_friction_angle"]
        )
        wall_back_face_angle_rad = np.radians(self.wall_back_face_angle)
        backfill_slope_angle_rad = np.radians(self.backfill_slope_angle)
        wall_interface_friction_angle_rad = np.radians(
            self.wall_interface_friction_angle
        )

        term1 = np.cos(drained_friction_angle_rad + wall_back_face_angle_rad) ** 2
        term2 = np.cos(wall_back_face_angle_rad) ** 2
        term3 = np.cos(wall_interface_friction_angle_rad - wall_back_face_angle_rad)
        term4 = np.sin(wall_interface_friction_angle_rad + drained_friction_angle_rad)
        term5 = np.sin(drained_friction_angle_rad + backfill_slope_angle_rad)
        term6 = term3
        term7 = np.cos(wall_back_face_angle_rad - backfill_slope_angle_rad)

        passive_coefficient = round(
            term1
            / (term2 * term3 * (1 - np.sqrt((term4 * term5) / (term6 * term7))) ** 2),
            3,
        )

        self.dataframe["coulomb_passive_coefficient"] = passive_coefficient

        return self.dataframe
        

    def calculate_top_active_pressure(self):
        """
        Calculates the top active pressure.
        """

        top_active_pressure = round(
            self.dataframe["coulomb_active_coefficient"]
            * self.dataframe["top_effective_stress"],
            1,
        )

        self.dataframe["coulomb_top_active_pressure"] = top_active_pressure

        return self.dataframe


    def calculate_bottom_active_pressure(self):
        """
        Calculates the bottom active pressure.
        """

        bottom_active_pressure = round(
            self.dataframe["coulomb_active_coefficient"]
            * self.dataframe["bottom_effective_stress"],
            1,
        )

        self.dataframe["coulomb_bottom_active_pressure"] = bottom_active_pressure

        return self.dataframe
        # self._dataframe["coulomb_bottom_active_pressure"] = round(
        #     self._dataframe["coulomb_active_coefficient"]
        #     * self._dataframe["bottom_effective_stress"],
        #     1,
        # )

    def calculate_top_passive_pressure(self):
        """
        Calculates the top passive pressure.
        """

        top_passive_pressure = round(
            self.dataframe["coulomb_passive_coefficient"]
            * self.dataframe["top_effective_stress"],
            1,
        )

        self.dataframe["coulomb_top_passive_pressure"] = top_passive_pressure

        return self.dataframe

        # self._dataframe["coulomb_top_passive_pressure"] = round(
        #     self._dataframe["coulomb_passive_coefficient"]
        #     * self._dataframe["top_effective_stress"],
        #     1,
        # )

    def calculate_bottom_passive_pressure(self):
        """
        Calculates the bottom passive pressure.
        """

        bottom_passive_pressure = round(
            self.dataframe["coulomb_passive_coefficient"]
            * self.dataframe["bottom_effective_stress"],
            1,
        )

        self.dataframe["coulomb_bottom_passive_pressure"] = bottom_passive_pressure

        return self.dataframe

        # self._dataframe["coulomb_bottom_passive_pressure"] = round(
        #     self._dataframe["coulomb_passive_coefficient"]
        #     * self._dataframe["bottom_effective_stress"],
        #     1,
        # )

    def calculate_coefficients(self):
        """
        Calculates both active and passive coefficients.
        """
        dataframe = self.calculate_active_coefficient()
        dataframe = self.calculate_passive_coefficient()
        return dataframe

    def calculate_all(self):
        """
        Calculates all the parameters including coefficients and pressures.
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

        # Get the soil layers _dataframe
        soil_layers = self._dataframe

        # Calculate maximum pressure for x-axis limit
        max_pressure = max(
            soil_layers["coulomb_top_active_pressure"].max(),
            soil_layers["coulomb_bottom_active_pressure"].max(),
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
                layer["coulomb_bottom_active_pressure"],
                layer["bottom_elevation"],
            )
            right_top = (layer["coulomb_top_active_pressure"], layer["top_elevation"])

            ka_value = layer["coulomb_active_coefficient"]
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
            plt.title("Coulomb Active Earth Pressures")

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

        # Get the soil layers _dataframe
        soil_layers = self._dataframe

        # Calculate maximum pressure for x-axis limit
        max_pressure = max(
            soil_layers["coulomb_top_passive_pressure"].max(),
            soil_layers["coulomb_bottom_passive_pressure"].max(),
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
                layer["coulomb_bottom_passive_pressure"],
                layer["bottom_elevation"],
            )
            right_top = (layer["coulomb_top_passive_pressure"], layer["top_elevation"])

            ka_value = layer["coulomb_passive_coefficient"]
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
            plt.title("Coulomb Passive Earth Pressures")

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