import numpy as np
import pandas as pd
from matplotlib.patches import Polygon
import matplotlib.pyplot as plt

from soil_profile import SoilProfile


class CoulombEarthPressure:
    def __init__(
        self,
        soil_profile,
        wall_top_elevation,
        wall_bottom_elevation,
        wall_backfill_angle,
        wall_back_face_angle,
        wall_interface_friction_angle,
    ):
        if not isinstance(soil_profile, SoilProfile):
            raise TypeError("soil_profile must be an instance of SoilProfile")
        if not isinstance(wall_top_elevation, (int, float)):
            raise TypeError("wall_top_elevation must be a number")
        if not isinstance(wall_bottom_elevation, (int, float)):
            raise TypeError("wall_bottom_elevation must be a number")
        if not isinstance(wall_backfill_angle, (int, float)):
            raise TypeError("wall_backfill_angle must be a number")
        if not isinstance(wall_back_face_angle, (int, float)):
            raise TypeError("wall_back_face_angle must be a number")
        if not isinstance(wall_interface_friction_angle, (int, float)):
            raise TypeError("wall_interface_friction_angle must be a number")

        self.soil_profile = soil_profile
        self.wall_top_elevation = wall_top_elevation
        self.wall_bottom_elevation = wall_bottom_elevation
        self.wall_backfill_angle = wall_backfill_angle
        self.wall_back_face_angle = wall_back_face_angle
        self.wall_interface_friction_angle = wall_interface_friction_angle

    def calculate_top_active_pressure(self):
        # Error checking
        if self.soil_profile is None:
            raise ValueError("soil_profile is not set")

    """
    Class representing Coulomb lateral earth pressure calculations.

    Attributes:
        soil_profile (object): Object representing the soil profile.
        wall_top_elevation (float): Elevation of the top of the wall.
        wall_bottom_elevation (float): Elevation of the bottom of the wall.
        wall_backfill_angle (float): Angle of the backfill slope.
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

    def calculate_active_coefficient(self):
        """
        Calculates the active coefficient of lateral earth pressure.
        """
        # Error checking
        if self.soil_profile is None:
            raise ValueError("soil_profile is not set")

        # Vectorized operations
        drained_friction_angle_rad = np.radians(
            self.soil_profile.dataframe["drained_friction_angle"]
        )
        wall_back_face_angle_rad = np.radians(self.wall_back_face_angle)
        wall_backfill_angle_rad = np.radians(self.wall_backfill_angle)
        wall_interface_friction_angle_rad = np.radians(
            self.wall_interface_friction_angle
        )

        term1 = np.cos(drained_friction_angle_rad - wall_back_face_angle_rad) ** 2
        term2 = np.cos(wall_back_face_angle_rad) ** 2
        term3 = np.cos(wall_interface_friction_angle_rad + wall_back_face_angle_rad)
        term4 = np.sin(wall_interface_friction_angle_rad + drained_friction_angle_rad)
        term5 = np.sin(drained_friction_angle_rad - wall_backfill_angle_rad)
        term6 = term3  # term3 is the same as term6
        term7 = np.cos(wall_back_face_angle_rad - wall_backfill_angle_rad)

        self.soil_profile.dataframe["coulomb_active_coefficient"] = round(
            term1
            / (term2 * term3 * (1 + np.sqrt((term4 * term5) / (term6 * term7))) ** 2),
            3,
        )

    def calculate_passive_coefficient(self):
        """
        Calculates the passive coefficient of lateral earth pressure.
        """
        # Error checking
        if self.soil_profile is None:
            raise ValueError("soil_profile is not set")
        # Vectorized operations
        drained_friction_angle_rad = np.radians(
            self.soil_profile.dataframe["drained_friction_angle"]
        )
        wall_back_face_angle_rad = np.radians(self.wall_back_face_angle)
        wall_backfill_angle_rad = np.radians(self.wall_backfill_angle)
        wall_interface_friction_angle_rad = np.radians(
            self.wall_interface_friction_angle
        )

        term1 = np.cos(drained_friction_angle_rad + wall_back_face_angle_rad) ** 2
        term2 = np.cos(wall_back_face_angle_rad) ** 2
        term3 = np.cos(wall_interface_friction_angle_rad - wall_back_face_angle_rad)
        term4 = np.sin(wall_interface_friction_angle_rad + drained_friction_angle_rad)
        term5 = np.sin(drained_friction_angle_rad + wall_backfill_angle_rad)
        term6 = term3
        term7 = np.cos(wall_back_face_angle_rad - wall_backfill_angle_rad)

        self.soil_profile.dataframe["coulomb_passive_coefficient"] = round(
            term1
            / (term2 * term3 * (1 - np.sqrt((term4 * term5) / (term6 * term7))) ** 2),
            3,
        )

    def calculate_top_active_pressure(self):
        """
        Calculates the top active pressure.
        """
        # Error checking
        if self.soil_profile is None:
            raise ValueError("soil_profile is not set")
        self.soil_profile.dataframe["coulomb_top_active_pressure"] = round(
            self.soil_profile.dataframe["coulomb_active_coefficient"]
            * self.soil_profile.dataframe["top_effective_stress"],
            1,
        )

    def calculate_bottom_active_pressure(self):
        """
        Calculates the bottom active pressure.
        """
        # Error checking
        if self.soil_profile is None:
            raise ValueError("soil_profile is not set")
        self.soil_profile.dataframe["coulomb_bottom_active_pressure"] = round(
            self.soil_profile.dataframe["coulomb_active_coefficient"]
            * self.soil_profile.dataframe["bottom_effective_stress"],
            1,
        )

    def calculate_top_passive_pressure(self):
        """
        Calculates the top passive pressure.
        """
        # Error checking
        if self.soil_profile is None:
            raise ValueError("soil_profile is not set")
        self.soil_profile.dataframe["coulomb_top_passive_pressure"] = round(
            self.soil_profile.dataframe["coulomb_passive_coefficient"]
            * self.soil_profile.dataframe["top_effective_stress"],
            1,
        )

    def calculate_bottom_passive_pressure(self):
        """
        Calculates the bottom passive pressure.
        """
        # Error checking
        if self.soil_profile is None:
            raise ValueError("soil_profile is not set")
        self.soil_profile.dataframe["coulomb_bottom_passive_pressure"] = round(
            self.soil_profile.dataframe["coulomb_passive_coefficient"]
            * self.soil_profile.dataframe["bottom_effective_stress"],
            1,
        )

    def calculate_coefficients(self):
        """
        Calculates both active and passive coefficients.
        """
        self.calculate_active_coefficient()
        self.calculate_passive_coefficient()

    def calculate_all(self):
        """
        Calculates all the parameters including coefficients and pressures.
        """
        self.calculate_active_coefficient()
        self.calculate_passive_coefficient()
        self.calculate_top_active_pressure()
        self.calculate_bottom_active_pressure()
        self.calculate_top_passive_pressure()
        self.calculate_bottom_passive_pressure()

    def plot_active_pressures(self):
        """
        Plots the soil layers with filled polygons representing the active pressure.

        Requires the 'soil_profile' attribute to be set with a valid dataframe.

        Returns:
            None
        """
        # Error checking
        if self.soil_profile is None:
            raise ValueError("soil_profile is not set")

        # Get the soil layers dataframe
        soil_layers = self.soil_profile.dataframe

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
            label = f"Layer {i+1}, Ka = {ka_value:.3f}"  # Updated label

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

        Requires the 'soil_profile' attribute to be set with a valid dataframe.

        Returns:
            None
        """
        # Error checking
        if self.soil_profile is None:
            raise ValueError("soil_profile is not set")

        # Get the soil layers dataframe
        soil_layers = self.soil_profile.dataframe

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
            label = f"Layer {i+1}, Kp = {ka_value:.3f}"  # Updated label

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
