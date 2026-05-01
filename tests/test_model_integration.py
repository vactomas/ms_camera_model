import pathlib
import unittest

from ms_camera_model import FilterSensorUnit
from ms_camera_model.filter_sensor import InterpolatedFilterSensorUnit

TEST_DIR = pathlib.Path(__file__).parent.resolve()


class IntegrationTests(unittest.TestCase):
    """ Basic integration tets for ms_camera_model """

    def test_filter_sensor_import(self):
        """ Test FilterSensorUnit import """

        fs_unit = FilterSensorUnit.from_excel(str(TEST_DIR / "mock_filter_sensor.xlsx"),
                                              str(TEST_DIR / "mock_filter_sensor.xlsx"))

        self.assertEqual(fs_unit.filter_spec.filter_transmittance[1, 1], 1)

        with self.assertRaises(ValueError):
            FilterSensorUnit.from_excel(str(TEST_DIR / "fail_mock_filter_sensor.xlsx"),
                                        str(TEST_DIR / "fail_mock_filter_sensor.xlsx"))

    def test_filter_sensor_interp(self):
        """ Test FilterSensorUnit interpolation to HS data """

        fs_unit = FilterSensorUnit.from_excel(str(TEST_DIR / "mock_filter_sensor.xlsx"),
                                              str(TEST_DIR / "mock_filter_sensor.xlsx"))

        hs_band_centers = []

        with self.assertRaises(ValueError):
            InterpolatedFilterSensorUnit.interpolate_to_hs_data(fs_unit, hs_band_centers)

        hs_band_centers = None

        with self.assertRaises(ValueError):
            InterpolatedFilterSensorUnit.interpolate_to_hs_data(fs_unit, hs_band_centers)

        hs_band_centers = [1, 1.5, 2]
        interpolated_fs_unit = InterpolatedFilterSensorUnit.interpolate_to_hs_data(fs_unit, hs_band_centers)
        interp_result = interpolated_fs_unit.filter_spec.filter_transmittance[1, 0]

        self.assertEqual(interp_result, 1.5, msg=f"Expected 1.5, got {interp_result}")


if __name__ == "__main__":
    unittest.main()
