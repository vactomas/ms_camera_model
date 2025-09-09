'''
=======================================================================================================================
Name:           Multispectral camera simulation model
Description:    Simulated model of a multispectral camera. Takes in hyperspectral data and colour filter specs. Outputs
                multispectral data
=======================================================================================================================
'''

from __future__ import annotations
import numpy as np
import spectral

from lib.spectral_scripts import vector_normalize
from lib.dataclasses import ImageData, FilterSpecs


class MultispectralCameraModel:
    """ Multispectral camera model

    :method load_hyperspectral_img:     Load hyperspectral data from image file and store it in own ImageData dataclass
    :method extract_img_data:           Extract from hyperspectral image data only frequencies passing through filters
    :method filter_img_data_matching:   Interpolate filter data and get values for frequencies matching hs data

    :param filters:                     List of FilterSpecs filter definitions
    :param hs_filename:                 Filename of the hyperspectral image file
    """

    def __init__(self, filters: list[FilterSpecs], hs_filename: str) -> None:
        self.hs_img_data: ImageData = self.load_hyperspectral_img(hs_filename)
        self.filters: list[FilterSpecs] = self.filter_img_data_matching(
            filters)
        self.ms_img_data: ImageData = self.extract_img_data()

    def load_hyperspectral_img(self, filename: str = 'test.hdr') -> ImageData:
        """ Load hyperspectral img data

        :param filename:    Filename of the hyperspectral image file
        """

        img = spectral.open_image(filename)
        return ImageData(vector_normalize(img.load()), img.bands.centers,
                         img.nbands)

    def extract_img_data(self) -> ImageData:
        """ Extract image data based on filters from hyperspectral data """

        ms_img_data = np.zeros_like(self.hs_img_data.img_data)

        for colour_filter in self.filters:
            ms_img_data += self.hs_img_data.img_data * colour_filter.filter_transmission[:,
                                                                                         1]

        return ImageData(ms_img_data, self.hs_img_data.band_centres,
                         self.hs_img_data.num_of_bands)

    def filter_img_data_matching(
            self, filters: list[FilterSpecs]) -> list[FilterSpecs]:
        """ Interpolate filter transmission data to match bands from hyperspectral img data 
    
        :param filters:     List of FilterSpecs filter definitions
        """

        corrected_filters = []

        for colour_filter in filters:
            interpolated_transmission = np.interp(
                self.hs_img_data.band_centres,
                colour_filter.filter_transmission[:, 0],
                colour_filter.filter_transmission[:, 1])
            corrected_filters.append(
                FilterSpecs(
                    np.column_stack([
                        self.hs_img_data.band_centres,
                        interpolated_transmission
                    ]), colour_filter.name, colour_filter.supplier,
                    colour_filter.band_center, colour_filter.band_width))

        return corrected_filters
