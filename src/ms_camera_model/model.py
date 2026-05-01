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

from ms_camera_model.errors import NoImageData, NoProvidedFilterSensorUnits
from ms_camera_model.filter_sensor import FilterSensorUnit, InterpolatedFilterSensorUnit
from ms_camera_model.image_data import (
    HyperspectralImageData,
    ImageData,
    ModeledMultispectralImageData,
)

logger = logging.getLogger(__name__)


@dataclass
class MultispectralCameraModel:
    """ Multispectral camera model 

    :param hs_data: HyperspectralImageData class instance
    :param filter_sensor_units: list of InterpolatedFilterSensorUnit class instances
    :param band_names: list of names for spectral bands
    :param out_data: simulated multispectral image data as ImageData class instance
    """
    hs_data: HyperspectralImageData
    filter_sensor_units: list[InterpolatedFilterSensorUnit]
    band_names: list[str]
    out_data: ModeledMultispectralImageData = field(init=False)

    @classmethod
    def create_model(cls, hs_data: HyperspectralImageData, fs_units: list[FilterSensorUnit],
                     band_names: list[str]) -> MultispectralCameraModel:
        """ Interpolate FilterSensorUnits to hyperspectral image data and create model with corrected units

        :param hs_data: HyperspectralImageData class instance
        :param fs_units: list of FilterSensorUnit class instances
        :param band_names: list of names for spectral bands
        """

        if not isinstance(fs_units, list):
            raise TypeError(f"Expected list of FilterSensorUnit class instances, got {type(fs_units)}")

        if not isinstance(hs_data, HyperspectralImageData):
            raise TypeError(f"Expected HyperspectralImageData, got {type(hs_data)}")

        if not isinstance(band_names, list):
            raise TypeError(f"Expected list of band names, got {type(band_names)}")

        if not fs_units:
            raise NoProvidedFilterSensorUnits

        corrected_units = []

        for filter_sensor_unit in fs_units:
            corrected_units.append(
                InterpolatedFilterSensorUnit.interpolate_to_hs_data(filter_sensor_unit, hs_data.band_centers))

        return MultispectralCameraModel(hs_data, corrected_units, band_names)

    def run_simulation(self) -> ImageData:
        """ Match filter_sensor_units specs with hs_data band_centers and calculate out_data 

        :return: Returns resulting simulated ImageData
        """
        logger.info("[MSModel] Units matched. Proceeding with data extraction...")
        self._extract_data()
        logger.info("[MSModel] Simulated multispectral data extraction completed")
        return self.out_data

    def _extract_data(self) -> None:
        """ Extract simulated multispectral data from the hyperspectral img_data using filter_sensor_units """

        if self.hs_data.img_data.size == 0:
            raise NoImageData

        shape = self.hs_data.img_data.shape
        modeled_ms_data = np.zeros((shape[0], shape[1], len(self.filter_sensor_units)), dtype=np.float32)
        modeled_ms_data_band_centers = []

        for i_unit, unit in enumerate(self.filter_sensor_units):
            modeled_ms_data[:, :, i_unit] = self._calculate_band(unit)
            modeled_ms_data_band_centers.append(unit.filter_spec.band_center)

        logger.info(f"[MSModel] modeled_ms_data max val: {modeled_ms_data.max()}")
        self.out_data = ModeledMultispectralImageData(modeled_ms_data, modeled_ms_data_band_centers,
                                                      len(self.filter_sensor_units), self.band_names)

    def _calculate_band(self, filter_sensor_unit: InterpolatedFilterSensorUnit) -> np.ndarray:
        """ Calculate the image created by a filter sensor unit 

        :param filter_sensor_unit: Input FilterSensorUnit
        """

        logger.info("[MSModel] Calculating single band image from hyperspectral data...")

        data_through_unit = self.hs_data.img_data * filter_sensor_unit.combined_response
        logger.info(f"[MSModel] Max data val: {data_through_unit.max()}")
        logger.info(f"[MSModel] Data type: {data_through_unit.dtype}")

        logger.info("[MSModel] Performing trapezoidal integration...")

        signal_integral = np.trapezoid(data_through_unit, axis=2)
        filter_integral = np.trapezoid(filter_sensor_unit.combined_response)

        out_img = signal_integral / filter_integral

        logger.info("[MSModel] Band calculation completed")

        return out_img
