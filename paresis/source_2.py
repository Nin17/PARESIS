"""Create sources
"""

import json
from typing import Any, Tuple, Union
import os
from os import PathLike
import warnings
from matplotlib.figure import Figure
import numpy as np
from numpy.typing import ArrayLike
import pandas as pd
import spekpy
import matplotlib.pyplot as plt
from scipy.constants import c, e, h
from utilities import Utility
from collections import namedtuple

# TODO combine sources
# TODO make spectrum a property
# tODO sort this shit out
# tODO make _spectrum list/array and then @property spectrum is Spectrum

class Spectrum:
    """_summary_
    """

    def __init__(self, energies, fluxes) -> None:
        """_summary_

        Parameters
        ----------
        energies : _type_
            _description_
        fluxes : _type_
            _description_
        """
        if len(energies) != len(fluxes):
            raise ValueError(f'energies and fluxes must have the same length. Currently {len(energies)} and {len(fluxes)} respectively')
        self._energies = energies
        self._fluxes = fluxes

    @property
    def energies(self):
        """_summary_

        Returns
        -------
        _type_
            _description_
        """
        return self._energies

    @energies.setter
    def energies(self, energies):
        """_summary_

        Parameters
        ----------
        energies : _type_
            _description_
        """
        self._energies = energies

    @property
    def fluxes(self):
        """_summary_

        Returns
        -------
        _type_
            _description_
        """
        return self._fluxes

    @fluxes.setter
    def fluxes(self, fluxes):
        """_summary_

        Parameters
        ----------
        fluxes : _type_
            _description_
        """
        self._fluxes = fluxes


class Source(Utility):
    """
    Class for x-ray sources
    """
    _spectrum = None

    def __init__(self, name: str = '', size: Union[float, ArrayLike] = 0.0,
                 shape: str = '', how: str = '',
                 spectrum: ArrayLike = None,
                 voltage: float = 0.0, target_material: str = '',
                 energy_sampling: float = 0.0,
                 exit_window_material: str = '',
                 exit_window_thickness: float = 0.0,  # Thickness in mm
                 json_file: str = 'json_files/sources.json',
                 csv_file: Union[str, PathLike] = '', **kwargs) -> None:
        """_summary_

        Parameters
        ----------
        name : str, optional
            _description_, by default 'source'
        size : Union[float, ArrayLike], optional
            _description_, by default 0.0
        shape : str, optional
            _description_, by default ''
        how : str, optional
            _description_, by default ''
        spectrum : ArrayLike, optional
            _description_, by default np.array(([1.0], [1.0]))
        voltage : float, optional
            _description_, by default 0.0
        target_material : str, optional
            _description_, by default ''
        energy_sampling : float, optional
            _description_, by default 1.0
        exit_window_material : str, optional
            _description_, by default ''
        exit_window_thickness : float, optional
            _description_, by default 0.0
        csv_file : Union[str, PathLike], optional
            _description_, by default ''
        """
        super().__init__(name, json_file)

        
        self.size = size
        self.shape = shape
        if spectrum:
            if not isinstance(spectrum, Spectrum):
                self._spectrum = Spectrum(*spectrum)
            else:
                self._spectrum = spectrum
        self.how = how
        self.voltage = voltage
        self.target_material = target_material
        self.exit_window_material = exit_window_material
        self.exit_window_thickness = exit_window_thickness
        self.energy_sampling = energy_sampling
        self.csv_file = csv_file
        self.json_file = json_file

        #   Get values from json if name provided
        if self.name:
            self._load()
        #TODO could set them after if they aren't the default values

        #   Get spectrum from spekpy
        if self.how == 'spekpy':
            self._spekpy_spectrum(**kwargs)

        #   Get spectrum from csv
        if self.csv_file and self.how != 'speckpy':
            self.from_csv()


    @property
    def spectrum(self) -> Spectrum:
        # if n
        #   Normalize intensities
        self._spectrum.fluxes /= self.total_flux

        #   Remove low energy data if its weighting is below 0.1%
        while self._spectrum.fluxes[0] < 1e-3:
            self._spectrum.energies = self._spectrum.energies[1:]
            self._spectrum.fluxes = self._spectrum.fluxes[1:]
            self._spectrum.fluxes /= self.total_flux
        return self._spectrum

        return Spectrum(np.array([1.0]), np.array([1.0]))

    @spectrum.setter
    def spectrum(self, spectrum: ArrayLike) -> None:
        self._spectrum = spectrum

    @property
    def total_flux(self) -> float:
        """
        Calculate the total flux for normalizing the spectrum

        Returns
        -------
        float
            Total flux in the spectrum
        """
        return sum(self._spectrum.fluxes)

    @property
    def wavenumbers(self) -> np.ndarray:
        """
        Calculate the wavenumbers present in the spectrum

        Returns
        -------
        np.ndarray
            1d array of wavenumbers
        """
        return 1e3*2*np.pi*self.spectrum[0]*e/(h*c)

    def _spekpy_spectrum(self, **kwargs) -> np.ndarray:
        """
        Obtain spectrum from spekpy

        Returns
        -------
        np.ndarray
            2d numpy array containing energies and fluxes
        """
        spectrum = spekpy.Spek(kvp=self.voltage, targ=self.target_material,
                               dk=self.energy_sampling, **kwargs)
        if self.exit_window_material and self.exit_window_thickness:
            print(self.exit_window_material, self.exit_window_thickness)
            try:
                spectrum.filter(self.exit_window_material,
                                self.exit_window_thickness)
            except Exception as error:
                raise error
        self._spectrum = Spectrum(*list(spectrum.get_spectrum()))

    def show_spectrum(self) -> Figure:
        """Plot the source spectrum

        Returns
        -------
        Figure
           matplotlib Figure object titled with the name
            and showing the spectrum
        """
        fig, axs = plt.subplots()
        axs.set_title(self.name)
        axs.set_xlabel('Energy (keV)')
        axs.set_ylabel('Normalized flux')
        axs.plot(self.spectrum.energies, self.spectrum.fluxes,
                 marker='o', markersize=2)
        return fig

    def from_csv(self) -> np.ndarray:
        """
        Get a source spectrum from a csv file

        Returns
        -------
        np.ndarray
            2d numpy array containing energies and fluxes
        """
        try:
            data = pd.read_csv(self.csv_file)
        except FileNotFoundError:
            if 'paresis' in self.csv_file:
                data = pd.read_csv(os.path.relpath(self.csv_file, 'paresis'))
            else:
                data = pd.read_csv(os.path.join('paresis', self.csv_file))

        energies = data['Energy'].to_numpy()
        fluxes = data['Flux'].to_numpy()
        self._spectrum = Spectrum(energies, fluxes)
