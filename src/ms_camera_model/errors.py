'''
Multispectral Camera Model - Custom Errors
==========================================

* **Description:** Custom error definitions to improve readability and make debugging easier
* **Author:** Tomas Vacek
'''


class ImageDataIncompatible(Exception):
    """ Image Data isn't incompatible """


class IncompatibleBandChoice(Exception):
    """ Chosen bands are not compatible with this method """


class NoProvidedFilepaths(Exception):
    """ No filepaths were provided """


class NoImageData(Exception):
    """ Provided ImageData instance doesn't contain img_data """


class NoProvidedArea(Exception):
    """ No area was provided """


class InvalidProvidedArea(Exception):
    """ Provided area is invalid """


class NoDarkFrame(Exception):
    """ No embedded dark frame found """
