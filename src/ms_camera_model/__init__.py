'''
Multispectral Camera Model - Simulation Model
=============================================

Simulated model of a multispectral camera. Takes in hyperspectral data and colour filter specs.
Outputs multispectral data
'''

__version__ = "0.1.0"

from . import data_comparison, image_visualiser
from .filter_sensor import FilterSensorUnit, FilterSpecs, SensorSpecs
from .image_data import (
    AreaLocation,
    HyperspectralImageData,
    ImageData,
    MultispectralImageData,
)
from .model import MultispectralCameraModel

__all__ = [
    "data_comparison", "image_visualiser", "FilterSensorUnit", "FilterSpecs", "SensorSpecs", "HyperspectralImageData",
    "ImageData", "MultispectralImageData", "AreaLocation", "MultispectralCameraModel"
]
