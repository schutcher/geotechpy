# constants.py

# Define constants for metric and imperial units
CONSTANTS = {
    "metric": {
        "water_unit_weight": 9.81,  # in kN/m³
        "atmospheric_pressure": 101.325,  # in kPa
        "length_units": "m",
        "force_units": "kN",
        "pressure_units": "kPa",
        "unit_weight_units": "kN/m³",

    },
    "imperial": {
        "water_unit_weight": 62.4,  # in lb/ft³
        "atmospheric_pressure": 2116.22,  # in psf
        "length_units": "ft",
        "force_units": "lbf",
        "pressure_units": "psf",
        "unit_weight_units": "pcf",
    }
}

def get_constants(units):
    """ Return constants based on the specified unit system. """
    return CONSTANTS.get(units.lower(), None)
