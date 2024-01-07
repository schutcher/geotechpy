import pandas as pd
import unittest
from calculations import calculate_unit_weight


class TestCalculateUnitWeight(unittest.TestCase):
    def setUp(self):
        self.dataframe = pd.DataFrame(
            {
                "saturated_unit_weight": [120, 130, 140],
                "moist_unit_weight": [110, 120, 130],
                "top_elevation": [5, 10, 15],
            }
        )
        self.water_surface_elev = 5

    def test_calculate_unit_weight_above_water(self):
        calculate_unit_weight(self.dataframe, self.water_surface_elev)
        expected_unit_weight = [120 - 62.4, 120, 130]
        actual_unit_weight = self.dataframe["unit_weight"].tolist()
        self.assertEqual(actual_unit_weight, expected_unit_weight)

    def test_calculate_unit_weight_below_water(self):
        calculate_unit_weight(self.dataframe, self.water_surface_elev)
        expected_unit_weight = [120 - 62.4, 120, 130]
        actual_unit_weight = self.dataframe["unit_weight"].tolist()
        self.assertEqual(actual_unit_weight, expected_unit_weight)

    def test_missing_columns(self):
        self.dataframe = pd.DataFrame(
            {"saturated_unit_weight": [120, 130, 140], "top_elevation": [5, 10, 15]}
        )
        with self.assertRaises(KeyError):
            calculate_unit_weight(self.dataframe, self.water_surface_elev)

    def test_unexpected_error(self):
        self.dataframe = pd.DataFrame(
            {
                "saturated_unit_weight": [120, 130, 140],
                "moist_unit_weight": [110, 120, 130],
                "top_elevation": [5, 10, 15],
            }
        )
        with self.assertRaises(Exception):
            calculate_unit_weight(self.dataframe, "not a number")


if __name__ == "__main__":
    unittest.main()
