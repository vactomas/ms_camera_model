"""
Multispectral Camera Model - Custom Errors
==========================================

* **Description:** Custom error definitions to improve readability and make debugging easier
* **Author:** Tomas Vacek
* **Year:** 2026
* **License:** MIT License
"""


class ImageDataIncompatible(Exception):
    """ Image Data is incompatible """


class IncompatibleBandChoice(Exception):
    """ Chosen bands are not compatible with this method """


class NoProvidedFilepaths(Exception):
    """ No filepaths were provided """


class NoImageData(Exception):
    """ Provided ImageData instance doesn't contain img_data """


class NoProvidedArea(Exception):
    """ No area was provided """


class NoBandCenters(Exception):
    """ Provided ImageData class instance doesn't contain band_centers """


class InvalidProvidedArea(Exception):
    """ Provided area is invalid """


class NoDarkFrame(Exception):
    """ No embedded dark frame found """


class AreaOutsideOfBounds(Exception):
    """ The provided area is outside of bounds of the image """


class ImageImportFailed(Exception):
    """ Image import failed """


class WavelengthMismatch(Exception):
    """ Provided filter wavelengths do not match with available hyperspectral image data """


class NoProvidedFilterSensorUnits(Exception):
    """ No provided FilterSensorUnit class instances """


class ImageRegistrationFailed(Exception):
    """ Image registration is not possible for the provided ImageData class instances """
