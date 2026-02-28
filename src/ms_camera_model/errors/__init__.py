'''
=======================================================================================================================
- Name:         Multispectral camera model - Custom errors
- Description:  Custom error definitions to improve readability and make debugging easier
- Author:       Tomas Vacek
=======================================================================================================================
'''

__all__ = ["ImageDataIncompatible", "IncompatibleBandChoice", "NoImageData", "NoProvidedArea", "NoProvidedFilepaths"]

from ms_camera_model.errors.errors import (
    ImageDataIncompatible,
    IncompatibleBandChoice,
    NoImageData,
    NoProvidedArea,
    NoProvidedFilepaths,
)
