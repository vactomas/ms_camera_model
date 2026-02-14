'''
=======================================================================================================================
- Name:         Multispectral camera model - simulation model
- Description:  Simulated model of a multispectral camera. Takes in hyperspectral data and colour filter specs. Outputs
                multispectral data
- Author:       Tomas Vacek
=======================================================================================================================
'''

from __future__ import annotations
import numpy as np

from dataclasses import dataclass, field

import logging
logger = logging.getLogger(__name__)

from ms_camera_model.image_data import HyperspectralImageData, ImageData
from ms_camera_model.filter_sensor import FilterSensorUnit, FilterSpecs, SensorSpecs
from ms_camera_model.errors import NoImageData


@dataclass
class MultispectralCameraModel:
    """ Multispectral camera model 

    :param hs_data: HyperspectralImageData class instance
    :param filter_sensor_units: list of FilterSensorUnit class instances
    :param out_data: simulated multispectral image data as ImageData class instance
    """
    hs_data: HyperspectralImageData
    filter_sensor_units: list[FilterSensorUnit]
    out_data: ImageData = field(init=False)

    def __post_init__(self) -> None:
        """ Match filter_sensor_units specs with hs_data band_centers and calculate out_data """
        logger.info("[MSModel] Creating MultispectralCameraModel instance, proceeding with filter_sensor_units matching to provided hyperspectral data...")
        self._filter_sensor_data_matching()
        logger.info("[MSModel] Units matched. Proceeding with data extraction...")
        self._extract_data()
        logger.info("[MSModel] Simulated multispectral data extraction completed")

    def _extract_data(self) -> None:
        """ Extract simulated multispectral data from the hyperspectral img_data using filter_sensor_units """

        if self.hs_data.img_data.size == 0:
            raise NoImageData
        
        self.hs_data.img_data = np.rot90(self.hs_data.img_data, k=1, axes=(0, 1))

        shape = self.hs_data.img_data.shape
        modeled_ms_data = np.zeros((shape[0], shape[1], len(self.filter_sensor_units)))
        modeled_ms_data_band_centers = []

        for i_unit in range(len(self.filter_sensor_units)):
            self.filter_sensor_units[i_unit].calculate_combined_attenuation()
            logger.info(f"[MSModel] FS_{i_unit} max combined attenuation is {self.filter_sensor_units[i_unit].combined_attenuation.max()}")
            modeled_ms_data[:, :, i_unit] = self._calculate_band(self.filter_sensor_units[i_unit])
            modeled_ms_data_band_centers.append(self.filter_sensor_units[i_unit].filter_spec.band_center)

        logger.info(f"[MSModel] modeled_ms_data max val: {modeled_ms_data.max()}")
        self.out_data = ImageData(modeled_ms_data, modeled_ms_data_band_centers, len(self.filter_sensor_units))

    def _filter_sensor_data_matching(self) -> None:
        """ Interpolate filter transmission data to match bands from hyperspectral img data """

        corrected_units = []

        for filter_sensor_unit in self.filter_sensor_units:
            
            filter_interp = np.interp(
                    self.hs_data.band_centers,
                    filter_sensor_unit.filter_spec.filter_transmission[:, 0],
                    filter_sensor_unit.filter_spec.filter_transmission[:, 1])
            sensor_interp = np.interp(
                    self.hs_data.band_centers,
                    filter_sensor_unit.sensor_spec.sensor_qe_curve[:, 0],
                    filter_sensor_unit.sensor_spec.sensor_qe_curve[:, 1])

            corrected_units.append(
                FilterSensorUnit(
                    FilterSpecs(
                        np.column_stack([
                            self.hs_data.band_centers,
                            filter_interp
                        ]),
                        filter_sensor_unit.filter_spec.name,
                        filter_sensor_unit.filter_spec.supplier,
                        filter_sensor_unit.filter_spec.band_center,
                        filter_sensor_unit.filter_spec.band_width
                    ),
                    SensorSpecs(
                        np.column_stack([
                            self.hs_data.band_centers,
                            sensor_interp
                        ]),
                        filter_sensor_unit.sensor_spec.name,
                        filter_sensor_unit.sensor_spec.supplier,
                        filter_sensor_unit.sensor_spec.sensor_type
                    ),
                    np.array([])
                )
            )

        self.filter_sensor_units = corrected_units


    def _calculate_band(self, filter_sensor_unit: FilterSensorUnit) -> np.ndarray:
        """ Calculate the image created by a filter sensor unit """

        logger.info("[MSModel] Calculating single band image from hyperspectral data...")

        data_through_unit = self.hs_data.img_data * filter_sensor_unit.combined_attenuation
        logger.info(f"[MSModel] Max data val: {data_through_unit.max()}")
        logger.info(f"[MSModel] Data type: {data_through_unit.dtype}")

        logger.info("[MSModel] Performing trapezoidal integration...")

        out_img = np.trapezoid(data_through_unit, axis=2)

        logger.info(f"[MSModel] Max integrated val: {out_img.max()}")
        logger.info(f"[MSModel] Data type: {out_img.dtype}")

        out_img = out_img / data_through_unit.shape[2]

        logger.info(f"[MSModel] Max out_img val: {out_img.max()}")
        logger.info(f"[MSModel] Data type: {out_img.dtype}")

        logger.info("[MSModel] Band calculation completed")

        return out_img
