"""
Multispectral Camera Model - Simulation Model
=============================================

* **Description:** Simulated model of a multispectral camera. Takes in hyperspectral data and colour filter specs.
Outputs multispectral data
* **Author:** Tomas Vacek
* **Year:** 2026
* **License:** MIT License
"""

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
from ms_camera_model.operations.interpolation import interpolate_light_data
from ms_camera_model.schemas.enums import SimulationMode
from ms_camera_model.schemas.light import LightSourceSpec

logger = logging.getLogger(__name__)


@dataclass
class MultispectralCameraModel:
    """ Multispectral camera model 

    :param hs_data: HyperspectralImageData class instance
    :param filter_sensor_units: list of InterpolatedFilterSensorUnit class instances
    :param band_names: list of names for spectral bands
    :param lightsource_spec: LightsourceSpec class instance
    :param simulation_mode: SimulationMode selection, defaults to reflectance
    :param out_data: simulated multispectral image data as ImageData class instance

    :method create_model: alternative constructor ensuring FilterSensorUnit interpolation takes place
    :method run_simulation: run simulation of the multispectral camera
    """
    hs_data: HyperspectralImageData
    filter_sensor_units: list[InterpolatedFilterSensorUnit]
    band_names: list[str]
    lightsource_spec: LightSourceSpec | None = None
    simulation_mode: SimulationMode = SimulationMode.REFLECTANCE

    out_data: ModeledMultispectralImageData = field(init=False)

    def __post_init__(self) -> None:
        """ Post init type check of InterpolatedFilterSensorUnit """
        if not all(isinstance(item, InterpolatedFilterSensorUnit) for item in self.filter_sensor_units):
            raise TypeError("Provided filter_sensor_units are not all of type InterpolatedFilterSensorUnit")

    @classmethod
    def create_model(cls,
                     hs_data: HyperspectralImageData,
                     fs_units: list[FilterSensorUnit],
                     band_names: list[str],
                     lightsource_spec: LightSourceSpec | None = None,
                     simulation_mode: SimulationMode = SimulationMode.REFLECTANCE) -> MultispectralCameraModel:
        """ Interpolate FilterSensorUnits to hyperspectral image data and create model with corrected units

        :param hs_data: HyperspectralImageData class instance
        :param fs_units: list of FilterSensorUnit class instances
        :param band_names: list of names for spectral bands
        :param lightsource_spec: LightsourceSpec class instance
        :param simulation_mode: defines simulation mode ('reflectance' or 'radiance')
        :return: MultispectralCameraModel
        :raises TypeError: if fs_units are not a list
        :raises TypeError: if hs_data is not an instance of HyperspectralImageData
        :raises TypeError: if band_names are not a list
        :raises TypeError: if lightsource_spec is not an instance of LightsourceSpec
        :raises ValueError: if selected simulation_mode is RADIANCE and lightsource_spec is empty
        :raises NoProvidedFilterSensorUnits: if no FilterSensorUnits are provided
        """

        if not isinstance(fs_units, list):
            raise TypeError(f"Expected list of FilterSensorUnit class instances, got {type(fs_units)}")

        if not isinstance(hs_data, HyperspectralImageData):
            raise TypeError(f"Expected HyperspectralImageData, got {type(hs_data)}")

        if not isinstance(band_names, list):
            raise TypeError(f"Expected list of band names, got {type(band_names)}")

        if not fs_units:
            raise NoProvidedFilterSensorUnits

        if simulation_mode == SimulationMode.RADIANCE:
            if not lightsource_spec:
                raise ValueError("Missing LightSourceSpec")
            if not isinstance(lightsource_spec, LightSourceSpec):
                raise TypeError(f"Expected LightSourceSpec, got {type(lightsource_spec)}")

            interpolated_lightsource_spec = interpolate_light_data(lightsource_spec, hs_data.band_centers)

        else:
            interpolated_lightsource_spec = None

        corrected_units = []

        for filter_sensor_unit in fs_units:
            corrected_units.append(
                InterpolatedFilterSensorUnit.interpolate_to_hs_data(filter_sensor_unit, hs_data.band_centers))

        return MultispectralCameraModel(hs_data, corrected_units, band_names, interpolated_lightsource_spec,
                                        simulation_mode)

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

        modeled_ms_data = np.clip(modeled_ms_data, a_min=0.0, a_max=None)
        self.out_data = ModeledMultispectralImageData(modeled_ms_data, modeled_ms_data_band_centers,
                                                      len(self.filter_sensor_units), self.band_names)

    def _calculate_band(self, filter_sensor_unit: InterpolatedFilterSensorUnit) -> np.ndarray:
        """ Calculate the image created by a filter sensor unit 

        :param filter_sensor_unit: Input FilterSensorUnit
        :return: np.ndarray of a single band
        """

        logger.info("[MSModel] Calculating single band image from hyperspectral data...")

        if self.simulation_mode == SimulationMode.REFLECTANCE:
            logger.info("[MSModel] Using reflectance simulation mode")
            data_through_unit = self.hs_data.img_data * filter_sensor_unit.combined_response
            filter_denominator = filter_sensor_unit.combined_response
        elif self.simulation_mode == SimulationMode.RADIANCE:
            logger.info("[MSModel] Using radiance simulation mode")
            data_through_unit = self.hs_data.img_data * filter_sensor_unit.combined_response * self.lightsource_spec.irradiance[:,
                                                                                                                                1]
            filter_denominator = filter_sensor_unit.combined_response * self.lightsource_spec.irradiance[:, 1]

        logger.info("[MSModel] Performing trapezoidal integration...")

        signal_integral = np.trapezoid(data_through_unit, self.hs_data.band_centers, axis=2)
        filter_integral = np.trapezoid(filter_denominator, self.hs_data.band_centers)

        out_img = signal_integral / filter_integral

        logger.info("[MSModel] Band calculation completed")

        return out_img
