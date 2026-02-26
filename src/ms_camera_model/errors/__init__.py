'''
=======================================================================================================================
- Name:         Multispectral camera model - Custom errors
- Description:  Custom error definitions to improve readability and make debugging easier
- Author:       Tomas Vacek
=======================================================================================================================
'''

__all__ = ["ImageDataIncompatible", "IncompatibleBandChoice", "NoProvidedFilepaths", "NoImageData"]

from ms_camera_model.errors.errors import ImageDataIncompatible
from ms_camera_model.errors.errors import IncompatibleBandChoice
from ms_camera_model.errors.errors import NoProvidedFilepaths
from ms_camera_model.errors.errors import NoImageData
