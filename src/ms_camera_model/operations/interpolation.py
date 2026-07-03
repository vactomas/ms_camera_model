"""
Multispectral Camera Model - Interpolation
==========================================

* **Description:** Interpolation of different components
* **Author:** Tomas Vacek
* **Year:** 2026
* **License:** MIT License
"""
import logging

import numpy as np

from ms_camera_model.schemas.light import LightSourceSpec

logger = logging.getLogger(__name__)


def interpolate_light_data(lightsource_spec: LightSourceSpec, hs_band_centers: list[float]) -> LightSourceSpec:
    """ Interpolate lightdata onto hyperspectral band centers

    :param lightsource_spec: specification of the lightsource
    :param hs_band_centers: list of hyperspectral band centers
    :return: LightSourceSpec
    """

    if not hs_band_centers:
        raise ValueError("Missing band center data for interpolation")
    if not isinstance(hs_band_centers, list):
        raise TypeError("Band centers are not a list")

    irradiance_interp = np.interp(hs_band_centers,
                                  lightsource_spec.irradiance[:, 0],
                                  lightsource_spec.irradiance[:, 1],
                                  left=0.0,
                                  right=0.0)
    return LightSourceSpec(name=lightsource_spec.name, irradiance=np.column_stack([hs_band_centers, irradiance_interp]))
