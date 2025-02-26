import pint

# Initialize a Unit Registry
ureg = pint.UnitRegistry()

# Length Conversion: Meters to Feet
length_in_meters = 10  # example value in meters
length_in_feet = (length_in_meters * ureg.meter).to(ureg.foot)
print(f"{length_in_meters} meters is {length_in_feet:.2f}")

# Force Conversion: Kilonewtons to Pounds-Force
force_in_kilonewtons = 5  # example value in kilonewtons
force_in_pounds_force = (force_in_kilonewtons * ureg.kilonewton).to(ureg.pound_force)
print(f"{force_in_kilonewtons} kilonewtons is {force_in_pounds_force:.2f}")

# Pressure Conversion: Kilopascals to Pounds per Square Foot
pressure_in_kilopascals = 100  # example value in kilopascals
pressure_in_psf = (pressure_in_kilopascals * ureg.kilopascal).to(
    ureg.pound_force / ureg.square_foot
)
print(f"{pressure_in_kilopascals} kilopascals is {pressure_in_psf:.2f}")

# Unit Weight Conversion: Kilonewtons per Cubic Meter to Pounds per Cubic Foot
unit_weight_in_kN_m3 = 20  # example value in kilonewtons per cubic meter
unit_weight_in_pcf = (unit_weight_in_kN_m3 * ureg.kilonewton / ureg.cubic_meter).to(
    ureg.pound_force / ureg.cubic_foot
)
print(f"{unit_weight_in_kN_m3} kN/m³ is {unit_weight_in_pcf:.2f}")
