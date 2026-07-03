"""
Multispectral Camera Model - Light source characteristic
========================================================

* **Description:** Definition of light source characteristic
* **Author:** Tomas Vacek
* **Year:** 2026
* **License:** MIT License
"""

import logging

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator

logger = logging.getLogger(__name__)


class LightSourceSpec(BaseModel):
    """ Light source characteristic definition

    :param name: name of the light source
    :param irradiance: spectral irradiance of the light source
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = "Generic"
    irradiance: np.ndarray = Field(frozen=True)

    @field_validator('irradiance', mode='after')
    @classmethod
    def validate_irradiance(cls, irradiance: np.ndarray) -> np.ndarray:
        if irradiance.ndim != 2 or irradiance.shape[1] != 2:
            raise ValueError(f"Expected 2 columns (wavelength, transmittance), got {np.shape(irradiance[1])}")
        return irradiance
