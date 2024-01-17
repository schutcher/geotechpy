import pandas as pd
import numpy as np
from numpy import radians, cos, sqrt, sin
from matplotlib.patches import Polygon
import matplotlib.pyplot as plt

from geotechpy.constants import get_constants

class LateralEarthPressure:
    """
    Calculates lateral earth pressure for a soil profile using the Coulomb or Rankine method.
    Args:
        dataframe (pandas.DataFrame): A dataframe containing the soil profile data.
        backfill_slope_angle (float): The angle of the backfill slope, in degrees.
        wall_back_face_angle (float): The angle of the wall back face, in degrees.
        wall_interface_friction_angle (float): The friction angle between the wall and the backfill, in degrees.
        units (str): The units system for the soil profile. Must be 'metric' or 'imperial'.

    Attributes:
        dataframe (pandas.DataFrame): A dataframe containing the soil profile data.
        backfill_slope_angle (float): The angle of the backfill slope, in degrees.
        wall_back_face_angle (float): The angle of the wall back face, in degrees.
        wall_interface_friction_angle (float): The friction angle between the wall and the backfill, in degrees.
        units (str): The units system for the soil profile. Must be 'metric' or 'imperial'.
        length_units (str): The length units for the soil profile. Either 'm' or 'ft'.
        pressure_units (str): The pressure units for the soil profile. Either 'kPa' or 'psf'.

    Raises:
        TypeError: If the dataframe is not a pandas DataFrame.
        TypeError: If the backfill_slope_angle is not a number.
        TypeError: If the wall_back_face_angle is not a number.
        TypeError: If the wall_interface_friction_angle is not a number.
        ValueError: If the backfill_slope_angle is greater than the minimum drained friction angle of the soil profile.
        ValueError: If the units are not 'metric' or 'imperial'.
    """
    def __init__(
        self,
        dataframe,
        backfill_slope_angle,
        wall_back_face_angle,
        wall_interface_friction_angle,
        units,
    ):
        self.dataframe = dataframe

        self.min_friction_angle = self.dataframe['drained_friction_angle'].min()

        self.backfill_slope_angle = backfill_slope_angle
        self.wall_back_face_angle = wall_back_face_angle
        self.wall_interface_friction_angle = wall_interface_friction_angle
        self.units = units

        # Retrieve the appropriate set of constants based on the specified units
        constants = get_constants(self.units)
        if constants is None:
            raise ValueError(f"Invalid units specified: {self.units}. Must be 'metric' or 'imperial'")

        # Assign the constants
        self.length_units = constants["length_units"]
        self.pressure_units = constants["pressure_units"] 


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

        if value > self.min_friction_angle:
            raise ValueError("Backfill slope angle cannot be greater than the minimum drained friction angle of the soil profile")

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

    def calculate_at_rest_coefficient(self, OCR=1):
        """
        Calculates the at-rest coefficient of lateral earth pressure and updates
        the 'at_rest_coefficient' column in the dataframe attribute of this instance.
        
        Based on guidance in the Caltrans Trenching and Shoring Manual

        Args:
            OCR (float): The overconsolidation ratio. Defaults to 1.

        Note: This method modifies the dataframe in-place.
        """
        phi = radians(self.dataframe["drained_friction_angle"])
        beta = radians(self.backfill_slope_angle)
        
        # For normally consolidated soils and vertical walls
        method1 = (1-sin(phi)) * (1-sin(beta))
        
        # For over consolidated soils, level backfill and vertical walls
        method2 = (1-sin(phi)) * OCR**(sin(phi))
        
        # If beta equals zero, use method 2, else use method 1
        at_rest_coefficient = method2 if beta == 0 else method1

        self.dataframe["at_rest_coefficient"] = round(at_rest_coefficient,2)
        return self.dataframe    
    
    def calculate_coulomb_active_coefficient(self):
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

        active_coefficient = term1 / (term2 * term3 * (1 + np.sqrt((term4 * term5) / (term6 * term7))) ** 2)

        self.dataframe["active_coefficient"] = round(active_coefficient,2)

        return self.dataframe
    
    
    def calculate_coulomb_passive_coefficient(self):
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

        passive_coefficient = term1 / (term2 * term3 * (1 - np.sqrt((term4 * term5) / (term6 * term7))) ** 2)
        
        self.dataframe["passive_coefficient"] = round(passive_coefficient,2)

        return self.dataframe 

    def calculate_coulomb_coefficients(self):
        """
        Calculates both active and passive coefficients.
        """
        dataframe = self.calculate_coulomb_active_coefficient()
        dataframe = self.calculate_coulomb_passive_coefficient()
        return dataframe     

    def calculate_rankine_active_coefficient(self):
        """
        Calculates the active coefficient of lateral earth pressure and updates 
        the 'rankine_active_coefficient' column in the dataframe attribute of this instance.

        The calculations do not account for  backface angle of the wall or wall friction.

        Returns:
            pandas.DataFrame: The updated dataframe with the active coefficient column.
        """
        # calculate active coefficient
        phi = radians(self.dataframe["drained_friction_angle"])
        beta = radians(self.backfill_slope_angle)
        numerator = cos(beta) - sqrt(
            (cos(beta)) ** 2
            - (cos(phi)) ** 2
        )
        denominator = cos(beta) + sqrt(
            (cos(beta)) ** 2
            - (cos(phi)) ** 2
        )
        active_coefficient = cos(beta) * numerator / denominator
        self.dataframe["active_coefficient"] = round(active_coefficient,2)
        return self.dataframe
    
    def calculate_rankine_passive_coefficient(self):
        """
        Calculates the passive coefficient of lateral earth pressure.

        Returns:
            pandas.DataFrame: The updated dataframe with the passive coefficient column.
        """
        # calculate passive coefficient
        phi = radians(
            self.dataframe["drained_friction_angle"]
        )
        beta = radians(self.backfill_slope_angle)
        numerator = cos(beta) + sqrt(
            (cos(beta)) ** 2
            - (cos(phi)) ** 2
        )
        denominator = cos(beta) - sqrt(
            (cos(beta)) ** 2
            - (cos(phi)) ** 2
        )
        passive_coefficient = cos(beta) * numerator / denominator
        self.dataframe["passive_coefficient"] = round(passive_coefficient,2)
        return self.dataframe     

    def calculate_rankine_coefficients(self):
        """
        Calculates both active and passive coefficients.
        """
        dataframe = self.calculate_rankine_active_coefficient()
        dataframe = self.calculate_rankine_passive_coefficient()
        return dataframe               

    def calculate_top_lateral_pressure(self, pressure_type):
        """
        Calculates the lateral earth pressure at the top of the soil layer based on the provided pressure type.
        
        Pressure types of "active", "at_rest" and "passive" are supported.
        
        Enforces a minimum pressure of zero.
        """

        # For active pressure, the sign is negative so that the cohesion term is subtracted;
        # For passive pressure, the sign is positive so that the cohesion term is added 
        sign = -1 if pressure_type == "active" else 1

        top_lateral_pressure = round(
            self.dataframe[f"{pressure_type}_coefficient"]
            * self.dataframe["top_effective_stress"]
            + sign * 2 * self.dataframe["drained_cohesion"] * sqrt(self.dataframe[f"{pressure_type}_coefficient"]),
            1,
        )

        top_at_rest_pressure = round(
            self.dataframe[f"{pressure_type}_coefficient"]
            * self.dataframe["top_effective_stress"],
            1,
        )

        top_pressure = top_lateral_pressure if pressure_type != "at_rest" else top_at_rest_pressure

        # Ensuring the pressure does not go below zero
        self.dataframe[f"top_{pressure_type}_pressure"] = top_pressure.apply(lambda p: max(p, 0))

        return self.dataframe



    def calculate_bottom_lateral_pressure(self, pressure_type):
        """
        Calculates the lateral earth pressure at the bottom of the soil layer based on the provided pressure type.
        Enforces a minimum pressure of zero.
        """

        # For active pressure, the sign is negative so that the cohesion term is subtracted;
        # For passive pressure, the sign is positive so that the cohesion term is added
        sign = -1 if pressure_type == "active" else 1

        bottom_lateral_pressure = round(
            self.dataframe[f"{pressure_type}_coefficient"]
            * self.dataframe["bottom_effective_stress"]
            + sign * 2 * self.dataframe["drained_cohesion"] * sqrt(self.dataframe[f"{pressure_type}_coefficient"]),
            1,
        )

        # Ensuring the pressure does not go below zero
        self.dataframe[f"bottom_{pressure_type}_pressure"] = bottom_lateral_pressure.apply(lambda p: max(p, 0))

        return self.dataframe
    
        # Calculate the active force for each layer
    def calculate_active_force(self):
        """
        Calculates the active force for each layer.
        """
        avg_active_pressure = (self.dataframe["top_active_pressure"] + self.dataframe["bottom_active_pressure"]) / 2
        layer_height = self.dataframe["layer_height"]
        active_force = avg_active_pressure * layer_height
        self.dataframe["active_force"] = round(active_force, 1)
        return self.dataframe
    
    # Calculate the passive force for each layer
    def calculate_passive_force(self):
        """
        Calculates the passive force for each layer.
        """
        avg_passive_pressure = (self.dataframe["top_passive_pressure"] + self.dataframe["bottom_passive_pressure"]) / 2
        layer_height = self.dataframe["layer_height"]
        passive_force = avg_passive_pressure * layer_height
        self.dataframe["passive_force"] = round(passive_force, 1)
        return self.dataframe
    
    # Calculate the at-rest force for each layer
    def calculate_at_rest_force(self):
        """
        Calculates the at-rest force for each layer.
        """
        avg_at_rest_pressure = (self.dataframe["top_at_rest_pressure"] + self.dataframe["bottom_at_rest_pressure"]) / 2
        layer_height = self.dataframe["layer_height"]
        at_rest_force = avg_at_rest_pressure * layer_height
        self.dataframe["at_rest_force"] = round(at_rest_force, 1)
        return self.dataframe
    
    # Calculate the water force for each layer
    def calculate_water_force(self):
        """
        Calculates the water force for each layer.
        """
        avg_water_pressure = (self.dataframe["top_water_pressure"] + self.dataframe["bottom_water_pressure"]) / 2
        layer_height = self.dataframe["layer_height"]
        water_force = avg_water_pressure * layer_height
        self.dataframe["water_force"] = round(water_force, 1)
        return self.dataframe
    
    def calculate_active_force_elevation(self):
        """
        Calculates the_elevation of the active force resultant for each layer.
        """
        layer_height = self.dataframe["layer_height"]
        bottom_elevation = self.dataframe["bottom_elevation"]
        top_pressure = self.dataframe["top_active_pressure"]
        bottom_pressure = self.dataframe["bottom_active_pressure"]
        relative_elevation = (bottom_pressure + 2 * top_pressure) * layer_height / (3 * (bottom_pressure + top_pressure))
        force_elevation = bottom_elevation + relative_elevation
        self.dataframe["active_force_elevation"] = round(force_elevation, 1)
        return self.dataframe
    
    def calculate_passive_force_elevation(self):
        """
        Calculates the_elevation of the passive force resultant for each layer.
        """
        layer_height = self.dataframe["layer_height"]
        bottom_elevation = self.dataframe["bottom_elevation"]
        top_pressure = self.dataframe["top_passive_pressure"]
        bottom_pressure = self.dataframe["bottom_passive_pressure"]
        relative_elevation = (bottom_pressure + 2 * top_pressure) * layer_height / (3 * (bottom_pressure + top_pressure))
        force_elevation = bottom_elevation + relative_elevation
        self.dataframe["passive_force_elevation"] = round(force_elevation, 1)
        return self.dataframe
    
    def calculate_at_rest_force_elevation(self):
        """
        Calculates the_elevation of the at-rest force resultant for each layer.
        """
        
        layer_height = self.dataframe["layer_height"]
        bottom_elevation = self.dataframe["bottom_elevation"]
        top_pressure = self.dataframe["top_at_rest_pressure"]
        bottom_pressure = self.dataframe["bottom_at_rest_pressure"]
        relative_elevation = (bottom_pressure + 2 * top_pressure) * layer_height / (3 * (bottom_pressure + top_pressure))
        force_elevation = bottom_elevation + relative_elevation
        self.dataframe["at_rest_force_elevation"] = round(force_elevation, 1)
        return self.dataframe
    
    def calculate_water_force_elevation(self):
        """
        Calculates the_elevation of the water force resultant for each layer.
        """
        layer_height = self.dataframe["layer_height"]
        bottom_elevation = self.dataframe["bottom_elevation"]
        top_pressure = self.dataframe["top_water_pressure"]
        bottom_pressure = self.dataframe["bottom_water_pressure"]
        relative_elevation = (bottom_pressure + 2 * top_pressure) * layer_height / (3 * (bottom_pressure + top_pressure))
        force_elevation = bottom_elevation + relative_elevation
        self.dataframe["water_force_elevation"] = round(force_elevation, 1)
        return self.dataframe
    
    def calculate_active_moment(self, wall_bottom_elevation):
        """
        Calculates the active moment about the wall base.
        """
        active_force = self.dataframe["active_force"]
        active_force_elevation = self.dataframe["active_force_elevation"]
        active_moment = active_force * (active_force_elevation - wall_bottom_elevation)
        self.dataframe["active_moment"] = round(active_moment, 1)
        return self.dataframe
    
    def calculate_passive_moment(self, wall_bottom_elevation):
        """
        Calculates the passive moment about the wall base.
        """
        passive_force = self.dataframe["passive_force"]
        passive_force_elevation = self.dataframe["passive_force_elevation"]
        passive_moment = passive_force * (passive_force_elevation - wall_bottom_elevation)
        self.dataframe["passive_moment"] = round(passive_moment, 1)
        return self.dataframe
    
    def calculate_at_rest_moment(self, wall_bottom_elevation):
        """
        Calculates the at-rest moment about the wall base.
        """
        at_rest_force = self.dataframe["at_rest_force"]
        at_rest_force_elevation = self.dataframe["at_rest_force_elevation"]
        at_rest_moment = at_rest_force * (at_rest_force_elevation - wall_bottom_elevation)
        self.dataframe["at_rest_moment"] = round(at_rest_moment, 1)
        return self.dataframe
    
    def calculate_water_moment(self, wall_bottom_elevation):
        """
        Calculates the water moment about the wall base.
        """
        water_force = self.dataframe["water_force"]
        water_force_elevation = self.dataframe["water_force_elevation"]
        water_moment = water_force * (water_force_elevation - wall_bottom_elevation)
        self.dataframe["water_moment"] = round(water_moment, 1)
        return self.dataframe
    
    def calculate_coulomb_active_all(self):
        """
        Calculates earth pressure coefficients and pressures using the Coulomb method.
        """
        dataframe = self.calculate_coulomb_active_coefficient()
        dataframe = self.calculate_top_lateral_pressure("active")
        dataframe = self.calculate_bottom_lateral_pressure("active")
        dataframe = self.calculate_active_force()
        dataframe = self.calculate_active_force_elevation()
        return dataframe

    def calculate_coulomb_passive_all(self):
        """
        Calculates earth pressure coefficients and pressures using the Coulomb method.
        """
        dataframe = self.calculate_coulomb_passive_coefficient()
        dataframe = self.calculate_top_lateral_pressure("passive")
        dataframe = self.calculate_bottom_lateral_pressure("passive")
        dataframe = self.calculate_passive_force()
        dataframe = self.calculate_passive_force_elevation()
        return dataframe   

    def calculate_rankine_active_all(self):
        """
        Calculates active earth pressure coefficients, pressures and forces using the Rankine method.
        """
        dataframe = self.calculate_rankine_active_coefficient()
        dataframe = self.calculate_top_lateral_pressure("active")
        dataframe = self.calculate_bottom_lateral_pressure("active")
        dataframe = self.calculate_active_force()
        dataframe = self.calculate_active_force_elevation()
        return dataframe 
    
    def calculate_rankine_passive_all(self):
        """
        Calculates passive earth pressure coefficients, pressures and forces using the Rankine method.
        """
        dataframe = self.calculate_rankine_passive_coefficient()
        dataframe = self.calculate_top_lateral_pressure("passive")
        dataframe = self.calculate_bottom_lateral_pressure("passive")
        dataframe = self.calculate_passive_force()
        dataframe = self.calculate_passive_force_elevation()
        return dataframe
    
    def calculate_at_rest_all(self, OCR):
        """
        Calculates earth pressure coefficients and pressures using the at-rest method.
        """
        dataframe = self.calculate_at_rest_coefficient(OCR)
        dataframe = self.calculate_top_lateral_pressure("at_rest")
        dataframe = self.calculate_bottom_lateral_pressure("at_rest")
        dataframe = self.calculate_at_rest_force()
        dataframe = self.calculate_at_rest_force_elevation()
        return dataframe
    
    def calculate_water_all(self):
        """
        Calculates water pressures and forces.
        """
        dataframe = self.calculate_water_force()
        dataframe = self.calculate_water_force_elevation()
        return dataframe
    




    def plot_lateral_pressures(self, method, pressure_type): # Method is either 'coulomb' or 'rankine'; pressure_type is either 'active' or 'passive'
        
        """
        Plots the soil layers with filled polygons representing the lateral earth pressure.
        """

        # Get the soil layers _dataframe
        soil_layers = self._dataframe

        # Calculate maximum pressure for x-axis limit
        max_pressure = max(
            soil_layers[f"top_{pressure_type}_pressure"].max(),
            soil_layers[f"bottom_{pressure_type}_pressure"].max(),
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
                layer[f"bottom_{pressure_type}_pressure"],
                layer["bottom_elevation"],
            )
            right_top = (layer[f"top_{pressure_type}_pressure"], layer["top_elevation"])

            k_value = layer[f"{pressure_type}_coefficient"]
            name = layer["layer_id"]
            label = f"Layer {name}, Kh = {k_value:.2f}"  # Updated label

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

        
        # Determine title layout
        # If pressure_type is 'at_rest', use title1
        title1 = f"{method.capitalize()} Earth Pressures"
        # Else, use title2
        title2 = f"{method.capitalize()} {pressure_type.capitalize()} Earth Pressures"
        title = title1 if pressure_type == "at_rest" else title2

        # Add title
        plt.title(title)

        # Set dynamic limits
        ax.set_xlim(0, max_pressure * 1.1)  # 10% more than max for better visibility
        ax.set_ylim(min_elevation, max_elevation)

        # Add labels, legend, and grid
        ax.set_xlabel(f"Stress ({self.pressure_units})")
        ax.set_ylabel(f"Elevation ({self.length_units})")
        ax.legend(title="Soil Layers")
        ax.grid(True)

        # Show the plot
        plt.show()

    def plot_at_rest_pressures(self):
        """
        Plots the soil layers with filled polygons representing the at-rest lateral earth pressure.
        """
        self.plot_lateral_pressures("At-rest", "at_rest")
    
    def plot_coulomb_active_pressures(self):
        """
        Plots the soil layers with filled polygons representing the active lateral earth pressure.
        """
        self.plot_lateral_pressures("coulomb", "active")

    def plot_coulomb_passive_pressures(self):
        """
        Plots the soil layers with filled polygons representing the passive lateral earth pressure.
        """
        self.plot_lateral_pressures("coulomb", "passive")

    def plot_rankine_active_pressures(self):
        """
        Plots the soil layers with filled polygons representing the active lateral earth pressure.
        """
        self.plot_lateral_pressures("rankine", "active")

    def plot_rankine_passive_pressures(self):
        """
        Plots the soil layers with filled polygons representing the passive lateral earth pressure.
        """
        self.plot_lateral_pressures("rankine", "passive")