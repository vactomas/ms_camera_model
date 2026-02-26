'''
=======================================================================================================================
- Name:         Multispectral camera model - Filters and Sensors
- Description:  Dataclasses and their methods for filters and sensors
- Author:       Tomas Vacek
=======================================================================================================================
'''

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class FilterSpecs:
    """ Filter specification """
    filter_transmission: np.ndarray
    name: str = "Generic"
    supplier: str = "Generic"
    band_center: int = 0
    band_width: int = 0


@dataclass
class SensorSpecs:
    """ Sensor specification """
    sensor_qe_curve: np.ndarray
    name: str = "Generic"
    supplier: str = "Generic"
    sensor_type: str = "CMOS"


@dataclass
class FilterSensorUnit:
    """ Combination of filter and sensor """
    filter_spec: FilterSpecs | None = None
    sensor_spec: SensorSpecs | None = None
    combined_attenuation: np.ndarray | None = None

    def load_filter_sensor(self, filename_filter: str,
                           filename_sensor: str) -> None:
        """ Load filter transmission and sensor spectral sensitivity from a file

        :param filename_filter: Filename or path to the filter xlsx file
        :param filename_sensor: Filename or path to the sensor xlsx file
        """

        logger.info(
            f"[FilterSensorUnit] Loading specifications\nFilter data: {filename_filter}\nSensor data: {filename_sensor}"
        )

        filter_data = pd.read_excel(filename_filter)
        sensor_data = pd.read_excel(filename_sensor)

        self.filter_spec = FilterSpecs(np.array(filter_data))
        self.sensor_spec = SensorSpecs(np.array(sensor_data))

    def calculate_combined_attenuation(self) -> None:
        """ Calculate combined attenuation of filter and sensor based on their transmission and qe (quantum efficiency) curve """

        logger.info("[FilterSensorUnit] Calculating combined attenuation")
        logger.info(
            f"[FilterSensorUnit] Filter transmission: {self.filter_spec.filter_transmission.shape}, Sensor QE curve: {self.sensor_spec.sensor_qe_curve.shape}"
        )

        self.combined_attenuation = self.filter_spec.filter_transmission[:,
                                                                         1] * self.sensor_spec.sensor_qe_curve[:,
                                                                                                               1]

        logger.info(
            f"[FilterSensorUnit] Combined attenuation: {self.combined_attenuation.shape}"
        )

    def interpolate_to_hs_data(
            self, hs_band_centers: list[float] | None) -> FilterSensorUnit:
        """ Interpolate the provided filter and sensor data to hyperspectral data """

        if hs_band_centers is None:
            raise ValueError(f"Missing band center data for interpolation")

        filter_interp = np.interp(hs_band_centers,
                                  self.filter_spec.transmission[:, 0],
                                  self.filter_spec.transmission[:, 1])
        sensor_interp = np.interp(hs_band_centers,
                                  self.sensor_spec.sensor_qe_curve[:, 0],
                                  self.sensor_spec.sensor_qe_curve[:, 1])

        interpolated_filter = FilterSpecs(
            np.column_stack([hs_band_centers, filter_interp]),
            self.filter_spec.name, self.filter_spec.supplier,
            self.filter_spec.band_center, self.filter_spec.band_width)

        interpolated_sensor = SensorSpecs(
            np.column_stack([hs_band_centers,
                             sensor_interp]), self.sensor_spec.name,
            self.sensor_spec.supplier, self.sensor_spec.sensor_type)

        return FilterSensorUnit(interpolated_filter, interpolated_sensor,
                                np.array([]))
