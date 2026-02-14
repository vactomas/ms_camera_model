'''
=======================================================================================================================
- Name:         Multispectral camera model - simulation model
- Description:  Simulated model of a multispectral camera. Takes in hyperspectral data and colour filter specs. Outputs
                multispectral data
- Author:       Tomas Vacek
=======================================================================================================================
'''

__version__ = "0.1.0"

__all__ = [
    "MultispectralCameraModel",
    "ImageData",
    "HyperspectralImageData",
    "MultispectralImageData",
    "FilterSpecs",
    "SensorSpecs",
    "FilterSensorUnit",
    "DataComparator"
]

from ms_camera_model.model import MultispectralCameraModel
from ms_camera_model.image_data import ImageData
from ms_camera_model.image_data import HyperspectralImageData
from ms_camera_model.image_data import MultispectralImageData
from ms_camera_model.filter_sensor import FilterSpecs
from ms_camera_model.filter_sensor import SensorSpecs
from ms_camera_model.filter_sensor import FilterSensorUnit
from ms_camera_model.data_comparison import DataComparator
