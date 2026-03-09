'''
Multispectral Camera Model - Simulation Model
=============================================

* **Description:** Simulated model of a multispectral camera. Takes in hyperspectral data and colour filter specs.
  Outputs multispectral data
* **Author:** Tomas Vacek
'''

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from ms_camera_model.errors import NoImageData
from ms_camera_model.filter_sensor import FilterSensorUnit
from ms_camera_model.image_data import HyperspectralImageData, ImageData

logger = logging.getLogger(__name__)


@dataclass
class MultispectralCameraModel:
    """ Multispectral camera model 

    :param hs_data: HyperspectralImageData class instance
    :param filter_sensor_units: list of FilterSensorUnit class instances
    :param out_data: simulated multispectral image data as ImageData class instance
    """
    hs_data: HyperspectralImageData
    filter_sensor_units: list[FilterSensorUnit]
    corrected_filter_sensor_units: list[FilterSensorUnit] = field(init=False)
    out_data: ImageData = field(init=False)

    def run_simulation(self) -> ImageData:
        """ Match filter_sensor_units specs with hs_data band_centers and calculate out_data 

        :return: Returns resulting simulated ImageData
        """
        logger.info(
            "[MSModel] Creating MultispectralCameraModel instance, proceeding with filter_sensor_units matching to provided hyperspectral data..."
        )
        self._filter_sensor_data_matching()
        logger.info("[MSModel] Units matched. Proceeding with data extraction...")
        self._extract_data()
        logger.info("[MSModel] Simulated multispectral data extraction completed")
        return self.out_data

    def _extract_data(self) -> None:
        """ Extract simulated multispectral data from the hyperspectral img_data using filter_sensor_units """

        if self.hs_data.img_data.size == 0:
            raise NoImageData

        shape = self.hs_data.img_data.shape
        modeled_ms_data = np.zeros((shape[0], shape[1], len(self.corrected_filter_sensor_units)))
        modeled_ms_data_band_centers = []

        for i_unit, unit in enumerate(self.corrected_filter_sensor_units):
            unit.calculate_combined_attenuation()
            logger.info(f"[MSModel] FS_{i_unit} max combined attenuation is {unit.combined_attenuation.max()}")
            modeled_ms_data[:, :, i_unit] = self._calculate_band(unit)
            modeled_ms_data_band_centers.append(unit.filter_spec.band_center)

        logger.info(f"[MSModel] modeled_ms_data max val: {modeled_ms_data.max()}")
        self.out_data = ImageData(modeled_ms_data, modeled_ms_data_band_centers,
                                  len(self.corrected_filter_sensor_units))

    def _filter_sensor_data_matching(self) -> None:
        """ Interpolate filter transmission data to match bands from hyperspectral img data """

        corrected_units = []

        for filter_sensor_unit in self.filter_sensor_units:
            corrected_units.append(filter_sensor_unit.interpolate_to_hs_data(self.hs_data.band_centers))

        self.corrected_filter_sensor_units = corrected_units

    def _calculate_band(self, filter_sensor_unit: FilterSensorUnit) -> np.ndarray:
        """ Calculate the image created by a filter sensor unit """

        logger.info("[MSModel] Calculating single band image from hyperspectral data...")

        data_through_unit = self.hs_data.img_data * filter_sensor_unit.combined_attenuation
        logger.info(f"[MSModel] Max data val: {data_through_unit.max()}")
        logger.info(f"[MSModel] Data type: {data_through_unit.dtype}")

        logger.info("[MSModel] Performing trapezoidal integration...")

        signal_integral = np.trapezoid(data_through_unit, axis=2)
        filter_integral = np.trapezoid(filter_sensor_unit.combined_attenuation)

        out_img = signal_integral / filter_integral

        logger.info("[MSModel] Band calculation completed")

        return out_img
