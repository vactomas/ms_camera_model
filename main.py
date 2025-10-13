'''
=======================================================================================================================
- Name:         Multispectral camera simulation model
- Description:  Simulated model of a multispectral camera. Takes in hyperspectral data and colour filter specs. Outputs
                multispectral data
- Author:       Tomas Vacek
=======================================================================================================================
'''

from __future__ import annotations
import numpy as np
import spectral

from lib.dataclasses import ImageData, FilterSpecs, SensorSpecs, FilterSensorUnit


class MultispectralCameraModel:
    """ Multispectral camera model

    :method load_hyperspectral_img:     Load hyperspectral data from image file and store it in own ImageData dataclass
    :method extract_img_data:           Extract from hyperspectral image data only frequencies passing through filters
    :method filter_img_data_matching:   Interpolate filter data and get values for frequencies matching hs data

    :param filter_sensor_units:         List of FilterSensorUnit definitions (sensor and it's corresponding filter)
    :param hs_filename:                 Filename of the hyperspectral image file
    """

    def __init__(self, filters_sensors: list[FilterSensorUnit], hs_filename: str) -> None:
        self.hs_img_data: ImageData = self.load_hyperspectral_img(hs_filename)
        self.filters_sensors: list[FilterSensorUnit] = self.filter_sensor_img_data_matching(
            filters_sensors)
        self.ms_img_data: ImageData = self.extract_img_data()

    def load_hyperspectral_img(self, filename: str = 'test.hdr') -> ImageData:
        """ Load hyperspectral img data

        :param filename:    Filename of the hyperspectral image file
        """
        img = spectral.open_image(filename)
        img_data = ImageData(img.load(), img.bands.centers, img.nbands)
        img_data.vector_normalize()
        return img_data

    def extract_img_data(self) -> ImageData:
        """ Extract image data based on filters from hyperspectral data """

        shape = self.hs_img_data.img_data.shape
        ms_img_data = np.zeros((shape[0], shape[1], len(self.filters_sensors)))
        filter_sensor_unit_num = 0

        for filter_sensor_unit in self.filters_sensors:
            filter_sensor_unit.calculate_combined_attenuation()
            ms_img_data[:, :, filter_sensor_unit_num] = self.calculate_output_of_filter_sensor_unit(filter_sensor_unit)
            filter_sensor_unit_num += 1

        return ImageData(ms_img_data, self.hs_img_data.band_centres,
                         self.hs_img_data.num_of_bands)

    def calculate_output_of_filter_sensor_unit(self, filter_sensor_unit: FilterSensorUnit) -> np.ndarray:
        """ Calculate the image created by a filter sensor unit """

        data_through_unit = self.hs_img_data.img_data * filter_sensor_unit.combined_attenuation
        output_image = np.trapezoid(data_through_unit, axis=2)

        return output_image

    def filter_sensor_img_data_matching(
            self, filters_sensors: list[FilterSensorUnit]) -> list[FilterSensorUnit]:
        """ Interpolate filter transmission data to match bands from hyperspectral img data 
    
        :param filters_sensors: List of FilterSensorUnit definitions
        """

        corrected_units = []

        for filter_sensor_unit in filters_sensors:
            
            filter_interp = np.interp(
                    self.hs_img_data.band_centres,
                    filter_sensor_unit.filter_spec.filter_transmission[:, 0],
                    filter_sensor_unit.filter_spec.filter_transmission[:, 1])
            sensor_interp = np.interp(
                    self.hs_img_data.band_centres,
                    filter_sensor_unit.sensor_spec.sensor_qe_curve[:, 0],
                    filter_sensor_unit.sensor_spec.sensor_qe_curve[:, 1])

            corrected_units.append(
                FilterSensorUnit(
                    FilterSpecs(
                        np.column_stack([
                            self.hs_img_data.band_centres,
                            filter_interp
                        ]),
                        filter_sensor_unit.filter_spec.name,
                        filter_sensor_unit.filter_spec.supplier,
                        filter_sensor_unit.filter_spec.band_center,
                        filter_sensor_unit.filter_spec.band_width
                    ),
                    SensorSpecs(
                        np.column_stack([
                            self.hs_img_data.band_centres,
                            sensor_interp
                        ]),
                        filter_sensor_unit.sensor_spec.name,
                        filter_sensor_unit.sensor_spec.supplier,
                        filter_sensor_unit.sensor_spec.sensor_type
                    ),
                    np.array([])
                )
            )

        return corrected_units
