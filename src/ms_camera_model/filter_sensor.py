'''
Multispectral Camera Model - Filters and Sensors
================================================

* **Description:** Dataclasses and their methods for filters and sensors
* **Author:** Tomas Vacek
'''

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ms_camera_model.errors import NoProvidedFilepaths, WavelengthMismatch

logger = logging.getLogger(__name__)


@dataclass
class FilterSpecs:
    """ Filter specification """
    filter_transmittance: np.ndarray
    name: str = "Generic"
    supplier: str = "Generic"
    band_center: float = 0
    band_width: float = 0

    def __post_init__(self) -> None:
        """ Post init checking to avoid empty filters """

        if not np.size(self.filter_transmittance):
            raise ValueError("Filter transmittance is an empty array")

        if self.filter_transmittance.ndim != 2 or self.filter_transmittance.shape[1] != 2:
            raise ValueError(
                f"Expected 2 columns (wavelength, transmittance), got {np.shape(self.filter_transmittance)[1]}")


@dataclass
class SensorSpecs:
    """ Sensor specification """
    sensor_qe_curve: np.ndarray
    name: str = "Generic"
    supplier: str = "Generic"
    sensor_type: str = "CMOS"

    def __post_init__(self) -> None:
        """ Post init checking to avoid empty qe curve """

        if not np.size(self.sensor_qe_curve):
            raise ValueError("Sensor QE curve is an empty array")

        if self.sensor_qe_curve.ndim != 2 or self.sensor_qe_curve.shape[1] != 2:
            raise ValueError(
                f"Expected 2 columns (wavelength, quantum_efficiency), got {np.shape(self.sensor_qe_curve)[1]}")


@dataclass
class FilterSensorUnit:
    """ Combination of filter and sensor """
    filter_spec: FilterSpecs
    sensor_spec: SensorSpecs
    combined_response: np.ndarray | None = None

    @classmethod
    def from_excel(cls, filename_filter: str, filename_sensor: str) -> FilterSensorUnit:
        """ Load filter transmission and sensor spectral sensitivity from Excel file

        :param filename_filter: Filename or path to the filter xlsx file
        :param filename_sensor: Filename or path to the sensor xlsx file
        """

        if not filename_filter or not filename_sensor:
            raise NoProvidedFilepaths

        logger.info(
            f"[FilterSensorUnit] Loading specifications\nFilter data: {filename_filter}\nSensor data: {filename_sensor}"
        )

        filter_data = pd.read_excel(filename_filter)
        sensor_data = pd.read_excel(filename_sensor)

        try:
            filter_values = filter_data.values.astype(np.float32)
            sensor_values = sensor_data.values.astype(np.float32)
        except ValueError as e:
            logger.error(f"[FilterSensorUnit] Filter or sensor Excel files contain non-numeric characters: {e}")
            raise ValueError from e

        filter_spec = FilterSpecs(np.array(filter_values))
        sensor_spec = SensorSpecs(np.array(sensor_values))

        return FilterSensorUnit(filter_spec, sensor_spec)

    def interpolate_to_hs_data(self, hs_band_centers: list[float]) -> FilterSensorUnit:
        """ Interpolate the provided filter and sensor data to hyperspectral data """

        logger.info("[FilterSensorUnit] Beginning FilterSensorUnit interpolation...")

        if not hs_band_centers:
            raise ValueError("Missing band center data for interpolation")

        min_hs_centers = min(hs_band_centers)
        max_hs_centers = max(hs_band_centers)

        active_response_mask = self.filter_spec.filter_transmittance[:, 1] > 0.01

        if not np.any(active_response_mask):
            raise ValueError(f"Filter {self.filter_spec.name} has no passband")

        active_response = self.filter_spec.filter_transmittance[active_response_mask]

        min_active_w = np.min(active_response[:, 0])
        max_active_w = np.max(active_response[:, 0])

        if min_active_w < min_hs_centers or max_active_w > max_hs_centers:
            raise WavelengthMismatch(
                "The defined filter has active response outside of available hyperspectral data wavelengths")

        filter_interp = np.interp(hs_band_centers,
                                  self.filter_spec.filter_transmittance[:, 0],
                                  self.filter_spec.filter_transmittance[:, 1],
                                  left=0.0,
                                  right=0.0)
        sensor_interp = np.interp(hs_band_centers,
                                  self.sensor_spec.sensor_qe_curve[:, 0],
                                  self.sensor_spec.sensor_qe_curve[:, 1],
                                  left=0.0,
                                  right=0.0)

        logger.info(f"[FilterSensorUnit] Filter interp {filter_interp.shape}")
        logger.info(f"[FilterSensorUnit] Sensor interp {sensor_interp.shape}")

        interpolated_filter = FilterSpecs(np.column_stack([hs_band_centers, filter_interp]), self.filter_spec.name,
                                          self.filter_spec.supplier, self.filter_spec.band_center,
                                          self.filter_spec.band_width)

        interpolated_sensor = SensorSpecs(np.column_stack([hs_band_centers, sensor_interp]), self.sensor_spec.name,
                                          self.sensor_spec.supplier, self.sensor_spec.sensor_type)

        interpolated_unit = FilterSensorUnit(interpolated_filter, interpolated_sensor)
        interpolated_unit.combined_response = filter_interp * sensor_interp

        return interpolated_unit
