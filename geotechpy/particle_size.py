from typing import Optional

soil_classification = {
    "MIT": {
        "Gravel": (2, None),
        "Sand": (0.06, 2),
        "Silt": (0.002, 0.06),
        "Clay": (None, 0.002),
    },
    "USDA": {
        "Gravel": (2, None),
        "Sand": (0.05, 2),
        "Silt": (0.002, 0.05),
        "Clay": (None, 0.002),
    },
    "AASHTO": {
        "Gravel": (2, 76.2),
        "Sand": (0.075, 2),
        "Silt": (0.002, 0.075),
        "Clay": (None, 0.002),
    },
    "USCS": {
        "Gravel": (4.75, 76.2),
        "Sand": (0.075, 4.75),
        "Fines": (None, 0.075),
    },
}


def classify_soil(grain_size: float, organization: str) -> Optional[str]:
    """
    Classify soil based on grain size and organization's classification criteria.

    Parameters:
    - grain_size (float): The size of the soil grain in mm.
    - organization (str): The organization whose criteria should be used for classification.

    Returns:
    - str: The soil type according to the specified organization's criteria.
    - None: If the organization is not found.
    """

    if organization not in soil_classification:
        return None

    for soil_type, (lower, upper) in soil_classification[organization].items():
        if lower is None:
            lower = float("-inf")
        if upper is None:
            upper = float("inf")
        if lower <= grain_size < upper:
            return soil_type
