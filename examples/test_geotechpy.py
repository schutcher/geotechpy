import pandas as pd

from geotechpy.soil_profile import SoilProfile
from geotechpy.lateral_earth_pressure import LateralEarthPressure

# from geotechpy.coulomb_earth_pressure import CoulombEarthPressure
# from geotechpy.rankine_earth_pressure import RankineEarthPressure


# Read the soil profile data from a CSV file
sp_import_df = pd.read_csv("soil_profile_Craig_Example_3_1.csv")

# Declare the units system for the soil profile
units = "metric"  # "imperial" or "metric"

# Declare the surcharge load
surcharge = 0

# Declare the water table elevation
water_table_elevation = 97

# Declare the backfill slope angle
backfill_slope_angle = 0

# Declare the wall back face angle, degrees from vertical
wall_back_face_angle = 0

# Declare the wall interface friction angle, degrees
wall_interface_friction_angle = 0

# Create an instance of the SoilProfile class
soil_profile = SoilProfile(sp_import_df, surcharge, water_table_elevation, units)
soil_profile.calculate_all()

print(soil_profile.dataframe)

soil_profile.plot_profile()

# soil_profile_df = soil_profile.dataframe

# lep = LateralEarthPressure(
#     soil_profile_df,
#     backfill_slope_angle,
#     wall_back_face_angle,
#     wall_interface_friction_angle,
#     units,
# )

# lep.calculate_coulomb_coefficients_pressures()

# lep.plot_coulomb_active_pressures()

# lep.plot_coulomb_passive_pressures()

# rep = LateralEarthPressure(
#     soil_profile_df,
#     backfill_slope_angle,
#     wall_back_face_angle,
#     wall_interface_friction_angle,
#     units,
# )

# rep.calculate_rankine_coefficients_pressures()

# rep.plot_rankine_active_pressures()

# rep.plot_rankine_passive_pressures()
