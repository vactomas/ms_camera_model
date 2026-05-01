import unittest

import numpy as np

from ms_camera_model import (
    AreaLocation,
    DataComparator,
    FilterSensorUnit,
    FilterSpecs,
    HyperspectralImageData,
    ImageData,
    MultispectralCameraModel,
    SensorSpecs,
)
from ms_camera_model.errors import (
    AreaOutsideOfBounds,
    ImageDataIncompatible,
    IncompatibleBandChoice,
    InvalidProvidedArea,
    NoProvidedArea,
    NoProvidedFilepaths,
    NoProvidedFilterSensorUnits,
    WavelengthMismatch,
)
from ms_camera_model.filter_sensor import InterpolatedFilterSensorUnit
from ms_camera_model.image_data import ModeledMultispectralImageData
from ms_camera_model.image_visualiser import ImageVisualiser


class TestModel(unittest.TestCase):
    """ Basic unit tests for ms_camera_model """

    def test_arealocation(self):
        """ Test AreaLocation check """

        with self.assertRaises(ValueError):
            AreaLocation(10, 5, 5, 10)

        with self.assertRaises(ValueError):
            AreaLocation(0, 2, 0, 2)

        with self.assertRaises(ValueError):
            AreaLocation(-5, 10, 5, 10)

        area_loc = AreaLocation(0, 0, 2, 2)
        area_loc_tuple = area_loc.as_tuple()

        self.assertEqual(area_loc_tuple, (0, 0, 2, 2), msg=f"Expected (0, 0, 2, 2), got {area_loc_tuple}")

    def test_sam(self):
        """ Test SAM """

        data_comp = DataComparator(None, None)

        fake_ms_ratios = np.array([5.0, 5.0, 5.0])
        fake_modeled_ms_ratios = np.array([5.0, 5.0, 5.0])

        angle = data_comp.calculate_spectral_angle_mapper(fake_ms_ratios, fake_modeled_ms_ratios)

        self.assertAlmostEqual(angle, 0.0, places=5, msg=f"Expected angle 0.0, got {angle}")

        fake_modeled_ms_ratios_zeros = np.array([0.0, 0.0, 0.0])

        with self.assertRaises(ValueError):
            data_comp.calculate_spectral_angle_mapper(fake_ms_ratios, fake_modeled_ms_ratios_zeros)

    def test_compare_band_ratios(self):
        """ Test compare_band_ratios function """

        img_data = np.ones((2, 2, 2))

        mock_hs_img_data = HyperspectralImageData(img_data, [1, 2], 2)
        mock_hs_img_data_more_bands = HyperspectralImageData(img_data, [1, 2], 5)
        mock_modeled_img_data = ModeledMultispectralImageData(img_data, [1, 2], 2, ["band1", "band2"])

        data_comp = DataComparator(mock_hs_img_data, mock_modeled_img_data)
        data_comp_invalid_nbands = DataComparator(mock_hs_img_data_more_bands, mock_modeled_img_data)

        area_loc = AreaLocation(0, 0, 2, 2)

        area_locations = [area_loc, area_loc]
        area_locations_invalid = [area_loc, area_loc, area_loc]

        with self.assertRaises(NoProvidedArea):
            data_comp.compare_band_ratios(None, None)

        with self.assertRaises(NoProvidedArea):
            data_comp.compare_band_ratios([], [])

        with self.assertRaises(InvalidProvidedArea):
            data_comp.compare_band_ratios(area_loc, area_locations)

        with self.assertRaises(InvalidProvidedArea):
            data_comp.compare_band_ratios(area_locations, area_loc)

        with self.assertRaises(ImageDataIncompatible):
            data_comp_invalid_nbands.compare_band_ratios(area_loc, area_loc)

        with self.assertRaises(InvalidProvidedArea):
            data_comp.compare_band_ratios(area_loc, area_loc, set_areas_globally=False)

        with self.assertRaises(InvalidProvidedArea):
            data_comp.compare_band_ratios(area_locations, area_locations_invalid)

        with self.assertRaises(InvalidProvidedArea):
            data_comp.compare_band_ratios(area_locations_invalid, area_locations)

        real_ratios, modeled_ratios = data_comp.compare_band_ratios(area_loc, area_loc)
        expected_ratios = np.array([0.5, 0.5])

        np.testing.assert_almost_equal(real_ratios, expected_ratios)
        np.testing.assert_almost_equal(modeled_ratios, expected_ratios)

    def test_mean_spectrum_area(self):
        """ Test the mean_spectrum_area """

        img_data = np.array([[1, 0], [2, 1]])

        with self.assertRaises(AreaOutsideOfBounds):
            ImageData.mean_spectrum_area(img_data, (5, 10, 5, 10))

    def test_empty_filter_specs(self):
        """ Test fail on empty filter transmittance init """

        with self.assertRaisesRegex(ValueError, "Filter transmittance"):
            filter_spec = FilterSpecs(np.array([[], []]))

        with self.assertRaisesRegex(ValueError, "Expected 2 columns"):
            filter_spec = FilterSpecs(np.array([[1, 2, 3], [1, 2, 3]]))

    def test_empty_sensor_specs(self):
        """ Test fail on empty sensor specs init """

        with self.assertRaisesRegex(ValueError, "Sensor QE"):
            sensor_spec = SensorSpecs(np.array([[], []]))

        with self.assertRaisesRegex(ValueError, "Expected 2 columns"):
            sensor_spec = SensorSpecs(np.array([[1, 2, 3], [1, 2, 3]]))

    def test_hs_data_interpolation(self):
        """ Test interpolation to hyperspectral image data """

        filter_spec = FilterSpecs(np.array([[1, 1], [2, 2]]))
        sensor_spec = SensorSpecs(np.array([[1, 1], [2, 2]]))

        filter_sensor_unit = FilterSensorUnit(filter_spec, sensor_spec)

        with self.assertRaisesRegex(ValueError, "Missing band center"):
            InterpolatedFilterSensorUnit.interpolate_to_hs_data(filter_sensor_unit, [])

        with self.assertRaises(WavelengthMismatch):
            InterpolatedFilterSensorUnit.interpolate_to_hs_data(filter_sensor_unit, [1000, 1001])

        filter_spec_no_pass = FilterSpecs(np.array([[1, 0], [2, 0]]))
        filter_sensor_unit_no_pass = FilterSensorUnit(filter_spec_no_pass, sensor_spec)

        with self.assertRaisesRegex(ValueError, "no passband"):
            InterpolatedFilterSensorUnit.interpolate_to_hs_data(filter_sensor_unit_no_pass, [1000, 1001])

    def test_filter_sensor_empty_import(self):
        """ Test importing FilterSensorUnit without providing file paths """

        with self.assertRaises(NoProvidedFilepaths):
            FilterSensorUnit.from_excel([], [])

    def test_imshow_incompatible_band_definition(self):
        """ Test plot with incompatible band definition """

        img_data = np.ones((2, 2, 2))

        mock_hs_img_data = HyperspectralImageData(img_data, [1, 2], 2)

        with self.assertRaises(IncompatibleBandChoice):
            ImageVisualiser.imshow(mock_hs_img_data, [2, 3, 4, 5])

        with self.assertRaisesRegex(TypeError, "Expected list"):
            ImageVisualiser.imshow(mock_hs_img_data, 2)

    def test_model_creation(self):
        """ Test creating the MultispectralCameraModel """

        img_data = np.ones((2, 2, 2))

        mock_hs_img_data = HyperspectralImageData(img_data, [1, 2], 2)

        filter_spec = FilterSpecs(np.array([[1, 1], [2, 2]]))
        sensor_spec = SensorSpecs(np.array([[1, 1], [2, 2]]))

        filter_sensor_unit = FilterSensorUnit(filter_spec, sensor_spec)

        with self.assertRaisesRegex(TypeError, "list of FilterSensorUnit"):
            MultispectralCameraModel.create_model(mock_hs_img_data, 2, [])

        with self.assertRaisesRegex(TypeError, "Expected HyperspectralImageData"):
            MultispectralCameraModel.create_model([], [filter_sensor_unit], [])

        with self.assertRaisesRegex(TypeError, "list of band names"):
            MultispectralCameraModel.create_model(mock_hs_img_data, [filter_sensor_unit], 2)

        with self.assertRaises(NoProvidedFilterSensorUnits):
            MultispectralCameraModel.create_model(mock_hs_img_data, [], [])


if __name__ == "__main__":
    unittest.main()
