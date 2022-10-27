"""Create detectors
"""
import os
from typing import Any, Tuple, Union
from xml.dom import minidom
import warnings
import numpy as np
from numpy.typing import ArrayLike
import numba as nb
from scipy.ndimage import gaussian_filter
import json
import inspect
# TODO make the scintillator a sample
# TODO some sorting out of loading and exceptions and stuff
# TODO could make a base class that has all the 
from utilities import Utility
from samples_2 import Uniform


class Detector(Utility):
    """_summary_
    """

    def __init__(self, name: str = '',
                 dimensions: ArrayLike = np.array([1.0, 1.0]),
                 pixel_size: ArrayLike = np.array([1.0, 1.0]),
                 margins: float = 0.0, energy_limit: float = 200.0,
                 psf: float = 0.0, efficiency_limit: float = 1.0,
                 bin_thresholds: Tuple = (), scintillator_material: str = '',
                 scintillator_thickness: float = 0.0, beta: Tuple = (),
                 json_file: str = 'json_files/detectors.json') -> None:
        """_summary_

        Parameters
        ----------
        name : str, optional
            _description_, by default ''
        dimensions : ArrayLike, optional
            _description_, by default np.array([1.0, 1.0])
        pixel_size : ArrayLike, optional
            _description_, by default np.array([1.0, 1.0])
        margins : float, optional
            _description_, by default 0.0
        energy_limit : float, optional
            _description_, by default 200.0
        psf : float, optional
            _description_, by default 0.0
        efficiency_limit : float, optional
            _description_, by default 1.0
        bin_thresholds : Tuple, optional
            _description_, by default ()
        scintillator_material : str, optional
            _description_, by default ''
        scintillator_thickness : float, optional
            _description_, by default 0.0
        beta : Tuple, optional
            _description_, by default ()
        json_file : str, optional
            _description_, by default 'json_files/detectors.json'
        """

        # self.name = name
        self.dimensions = dimensions
        self.pixel_size = pixel_size
        self.psf = psf
        self.efficiency_limit = efficiency_limit
        self.margins = margins
        self.energy_limit = energy_limit
        self.bin_thresholds = bin_thresholds
        self.scintillator_material = scintillator_material
        self.scintillator_thickness = scintillator_thickness
        self.beta = beta  # ??? why is this here
        # self.json_file = json_file
        super().__init__(name, json_file)

        #   Get values from xml if name provided
        if self.name:
            self._load()

        if not isinstance(self.pixel_size, np.ndarray):
            self.pixel_size = np.array([self.pixel_size]*2)

    # def __repr__(self) -> str:
    #     """
    #     String representation of the class

    #     Returns
    #     -------
    #     str
    #         The string representation of the class object
    #     """
    #     if self.__dict__.keys():
    #         i = max(map(len, list(self.__dict__.keys()))) + 1
    #         return '\n'.join([f'{k.rjust(i)}: {repr(v)}'
    #                           for k, v in sorted(self.__dict__.items())])
    #     return ''

    def detect(self, incident_wave: np.ndarray,
               effective_source_size: Union[float, np.ndarray],
               source_shape: str = 'circle') -> np.ndarray:
        """_summary_

        Parameters
        ----------
        incident_wave : np.ndarray
            _description_
        effective_source_size : Union[float, np.ndarray]
            _description_
        source_shape : str, optional
            _description_, by default 'circle'

        Returns
        -------
        np.ndarray
            _description_

        Raises
        ------
        NotImplementedError
            _description_
        """
        #   TODO different source shapes
        # incident_wave *= 100
        if effective_source_size and source_shape == 'circle':
            sigma_source = effective_source_size /\
                (2*np.sqrt(2*np.log(2)))
            incident_wave = gaussian_filter(
                incident_wave, sigma_source, mode='wrap')
        elif effective_source_size:
            raise NotImplementedError(
                'Source shapes other than a circle are not yet implemented')
        detected_image = resize(incident_wave, self.dimensions)
        if self.psf:
            detected_image = gaussian_filter(
                detected_image, self.psf, mode='wrap')
        random_state = np.random.default_rng()
        detected_image = random_state.poisson(detected_image)
        if self.margins:
            detected_image = detected_image[self.margins:-self.margins,
                                            self.margins:-self.margins]

        return detected_image


# @nb.njit(parallel=True)
def resize(image_to_resize: np.ndarray, size: np.ndarray) -> np.ndarray:
    """_summary_

    Parameters
    ----------
    image_to_resize : np.ndarray
        _description_
    size : np.ndarray
        _description_

    Returns
    -------
    np.ndarray
        _description_
    """
    size_y, size_x = size
    n_y, n_x = image_to_resize.shape
    if n_x == size_x and n_y == size_y:
        return image_to_resize
    #   TODO can probably do this all in numpy
    resized_image = np.ones((size_y, size_x))
    samp_factor = int(image_to_resize.shape[0]/size_y)

    for x_0 in range(size_y):
        for y_0 in range(size_x):
            resized_image[x_0, y_0] = \
                np.sum(image_to_resize[int(x_0*samp_factor):
                                       int(x_0*samp_factor+samp_factor),
                                       int(y_0*samp_factor):
                                       int(y_0*samp_factor+samp_factor)])

    return resized_image


    # def _from_xml(self) -> None:
    #     """
    #     Get and set the detector information from the xml file.

    #     Returns
    #     -------
    #     NoneType
    #         Return None after name found in xml else raise a warning and
    #                 return None
    #     """
    #     def get_text(node) -> Any:
    #         return node.childNodes[0].nodeValue
    #     try:
    #         xml_doc = minidom.parse(self.xml_file)
    #     except FileNotFoundError:
    #         if 'paresis' in self.xml_file:
    #             xml_doc = minidom.parse(
    #                 os.path.relpath(self.xml_file, 'paresis'))
    #         else:
    #             xml_doc = minidom.parse(os.path.join('paresis', self.xml_file))
    #     detectors = xml_doc.documentElement.getElementsByTagName('detector')
    #     names = [get_text(i.getElementsByTagName('name')[0]) for i in detectors]
    #     try:
    #         index = names.index(self.name)
    #     except ValueError:
    #         warnings.warn(f'name: {self.name} not found in xml file')
    #         return
    #     detector = detectors[index]
    #     for node in detector.childNodes:
    #         if node.localName:
    #             text = get_text(detector.getElementsByTagName(
    #                 node.localName)[0])
    #             if ',' in text:
    #                 text = np.array([float(i)
    #                                 for i in text.split(', ')])
    #             try:
    #                 if text.isdigit() or \
    #                         text.replace('.', '', 1).isdigit():
    #                     text = float(text)
    #             except AttributeError:
    #                 pass
    #             setattr(self, str(node.localName), text)
    #     return
