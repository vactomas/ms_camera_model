"""
Multispectral Camera Model - Filters and Sensors
================================================

* **Description:** Dataclasses and their methods for filters and sensors
* **Author:** Tomas Vacek
* **Year:** 2026
* **License:** MIT License
"""

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
        """ Post init checking to avoid empty filters 

        :raises ValueError: if filter_transmittance is an empty array
        :raises ValueError: if the filter_transmittance is not a 2D, 2 column array
        """

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
        """ Post init checking to avoid empty qe curve

        :raises ValueError: if sensor_qe_curve is an empty array
        :raises ValueError: if the sensor_qe_curve is not a 2D, 2 column array
        """

        if not np.size(self.sensor_qe_curve):
            raise ValueError("Sensor QE curve is an empty array")

        if self.sensor_qe_curve.ndim != 2 or self.sensor_qe_curve.shape[1] != 2:
            raise ValueError(
                f"Expected 2 columns (wavelength, quantum_efficiency), got {np.shape(self.sensor_qe_curve)[1]}")


@dataclass
class FilterSensorUnit:
    """ Combination of filter and sensor

    :param filter_spec: filter specification
    :param sensor_spec: sensor specification

    :method from_excel: alternative constructor which imports data from Excel
    """
    filter_spec: FilterSpecs
    sensor_spec: SensorSpecs

    @classmethod
    def from_excel(cls, filename_filter: str, filename_sensor: str) -> FilterSensorUnit:
        """ Load filter transmission and sensor spectral sensitivity from Excel file

        :param filename_filter: Filename or path to the filter xlsx file
        :param filename_sensor: Filename or path to the sensor xlsx file

        :return: FilterSensorUnit with loaded data
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


@dataclass
class InterpolatedFilterSensorUnit(FilterSensorUnit):
    """ FilterSensorUnit interpolated to hyperspectral data 

    :param combined_response: combined spectral response function for the filter and sensor

    :method interpolate_to_hs_data: alternative constructor, interpolates FilterSensorUnit to hyperspectral data
    """
    combined_response: np.ndarray | None = None

    @classmethod
    def interpolate_to_hs_data(cls, fs_unit: FilterSensorUnit,
                               hs_band_centers: list[float]) -> InterpolatedFilterSensorUnit:
        """ Interpolate the provided filter and sensor data to hyperspectral data 

        :param fs_unit: FilterSensorUnit class instance
        :param hs_band_centers: list[float] containing band centers of available hyperspectral data
        :return: InterpolatedFilterSensorUnit
        :raises ValueError: if hs_band_centers are None
        :raises ValueError: if the filter has no pass band
        :raises WavelengthMismatch: if the filter active response is outside of available hyperspectral wavelengths
        """

        logger.info("[InterpolatedFilterSensorUnit] Beginning FilterSensorUnit interpolation...")

        if not hs_band_centers:
            raise ValueError("Missing band center data for interpolation")

        min_hs_centers = min(hs_band_centers)
        max_hs_centers = max(hs_band_centers)

        active_response_mask = fs_unit.filter_spec.filter_transmittance[:, 1] > 0.01

        if not np.any(active_response_mask):
            raise ValueError(f"Filter {fs_unit.filter_spec.name} has no passband")

        active_response = fs_unit.filter_spec.filter_transmittance[active_response_mask]

        min_active_w = np.min(active_response[:, 0])
        max_active_w = np.max(active_response[:, 0])

        if min_active_w < min_hs_centers or max_active_w > max_hs_centers:
            raise WavelengthMismatch(
                "The defined filter has active response outside of available hyperspectral data wavelengths")

        filter_interp = np.interp(hs_band_centers,
                                  fs_unit.filter_spec.filter_transmittance[:, 0],
                                  fs_unit.filter_spec.filter_transmittance[:, 1],
                                  left=0.0,
                                  right=0.0)
        sensor_interp = np.interp(hs_band_centers,
                                  fs_unit.sensor_spec.sensor_qe_curve[:, 0],
                                  fs_unit.sensor_spec.sensor_qe_curve[:, 1],
                                  left=0.0,
                                  right=0.0)

        logger.info(f"[InterpolatedFilterSensorUnit] Filter interp {filter_interp.shape}")
        logger.info(f"[InterpolatedFilterSensorUnit] Sensor interp {sensor_interp.shape}")

        interpolated_filter = FilterSpecs(np.column_stack([hs_band_centers, filter_interp]), fs_unit.filter_spec.name,
                                          fs_unit.filter_spec.supplier, fs_unit.filter_spec.band_center,
                                          fs_unit.filter_spec.band_width)

        interpolated_sensor = SensorSpecs(np.column_stack([hs_band_centers, sensor_interp]), fs_unit.sensor_spec.name,
                                          fs_unit.sensor_spec.supplier, fs_unit.sensor_spec.sensor_type)

        combined_response = filter_interp * sensor_interp

        return InterpolatedFilterSensorUnit(interpolated_filter, interpolated_sensor, combined_response)
