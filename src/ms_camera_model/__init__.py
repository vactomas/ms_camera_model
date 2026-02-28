'''
Multispectral Camera Model - Simulation Model
=============================================

Simulated model of a multispectral camera. Takes in hyperspectral data and colour filter specs.
Outputs multispectral data
'''

__version__ = "0.1.0"

from .data_comparison import DataComparator
from .filter_sensor import FilterSensorUnit, FilterSpecs, SensorSpecs
from .image_data import (
    HyperspectralImageData,
    ImageData,
    MultispectralImageData,
)
from .model import MultispectralCameraModel

__all__ = [
    "DataComparator", "FilterSensorUnit", "FilterSpecs", "SensorSpecs", "HyperspectralImageData", "ImageData",
    "MultispectralImageData", "MultispectralCameraModel"
]
