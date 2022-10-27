"""_summary_
"""
from abc import ABC, abstractmethod
from typing import Tuple, Type, Union, Dict
import warnings
import time
from matplotlib.figure import Figure
import xraylib as xrl
import numpy as np
from numpy.typing import ArrayLike
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.ndimage import rotate, affine_transform
from scipy.constants import golden_ratio, c, e, h
from utilities import Utility
# from membranes_2 import Membrane2D, MultiLayerMembrane
# TODO move membranes to this file to fix stuff
# TODO all plots in same figure have the same colormap
# TODO oversampling factor that can be updated and changes the properties accordingly
# TODO use opencl
# import numba as nb
#   TODO 1ex doesn't work yet
# TODO define __add__ method
# TODO make another class where you can add different material objects together
# TODO make thickness a dictionary
# TODO update type hinting
# TODO probs move init to function rather than calculating all that there
# TODO check that transmission is actually correct
# TODO change to np.add.outer as much faster
# TODO make args lists that can have same number of elements as number of materials
# tODO show methods to have material arg or all if None
# TODO check pure phase and pure attenuation is correct
# TODO make gratings
# tODO oversampling parameter that changes it accordingly
# TODO materials and density better
# TODO dictionary of materials and densities
# FIXME lots of numerical errors in transmit_fresnel


def wavenumber(energy):
    """_summary_

    Parameters
    ----------
    energy : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    return 2*np.pi*energy*e*1e3/(c*h)


# TODO sort out initialization so that plots work
class Sample(Utility, ABC):
    """_summary_

    Parameters
    ----------
    ABC : _type_
        _description_
    """

    def __init__(self, name: str = '', materials: ArrayLike = (None,),
                 density: float = None,
                 dimensions: ArrayLike = np.array([1, 1]),
                 position: ArrayLike = np.array([0.0, 0.0]),
                 angle: float = 0.0,
                 pixel_size: float = 1.0,
                 pure_attenuation: bool = False,
                 pure_phase: bool = False,
                 material_file: str = 'Samples/DeltaBeta/Materials.csv',
                 json_file: str = 'json_files/samples.json') -> None:
        """_summary_

        Parameters
        ----------
        name : str, optional
            _description_, by default ''
        material : str, optional
            _description_, by default ''
        density : float, optional
            _description_, by default 1.0
        dimensions : np.ndarray, optional
            _description_, by default np.array([1.0, 1.0])
        pixel_size : float, optional
            _description_, by default 1.0
        """
        # self.name = name
        super().__init__(name, json_file)
        if name:
            self._load()
        self.materials = materials
        self.density = density
        self.dimensions = dimensions
        self.position = position
        self.angle = angle
        self.pixel_size = pixel_size
        self.pure_attenuation = pure_attenuation
        self.pure_phase = pure_phase
        self.material_file = material_file


    @property
    @abstractmethod
    def thickness(self) -> None:
        """_summary_

        Raises
        ------
        NotImplementedError
            _description_
        """
        raise NotImplementedError('Must overwrite thickness')

    # ??? is this necessary?
    @property
    @abstractmethod
    def max_thickness(self) -> None:
        """_summary_

        Raises
        ------
        NotImplementedError
            _description_
        """
        raise NotImplementedError('Must overwrite thickness')

    @property
    @abstractmethod
    def gradient(self) -> None:
        """_summary_

        Returns
        -------
        _type_
            _description_

        Raises
        ------
        NotImplementedError
            _description_
        """
        raise NotImplementedError('Must overwrite gradient')

    # TODO make it dictionary of refractive indicies

    def refractive_index(self, energy: float) -> np.ndarray:
        """_summary_

        Parameters
        ----------
        energies : np.ndarray
            _description_

        Returns
        -------
        np.ndarray
            _description_
        """
        data = pd.read_csv(self.material_file)
        data = data.set_index('Material')

        def _formula_handler(data, material):
            try:
                return data['Formula'][material]
            except KeyError:
                return material

        def _density_handler(data, material):
            if self.density is not None:
                return self.density
            else:
                try:
                    return data['Density'][material]
                except KeyError as error:
                    raise ValueError(
                        f'{material} is not in {self.material_file} and density is None') from error

        refractive_index = {material: xrl.Refractive_Index(_formula_handler(
            data, material), energy, _density_handler(data, material)) for material in self.materials}

        if self.pure_phase and self.pure_attenuation:
            raise ValueError(
                'Ambiguous as both self.pure_phase and self.pure_attenuation are True')
        if self.pure_phase:
            return {k: np.real(v) for k, v in refractive_index.items()}
        if self.pure_attenuation:
            return {k: 1+1j*np.imag(v) for k, v in refractive_index.items()}
        return refractive_index

    @abstractmethod
    def transmit_fresnel(self, incident_wave: np.ndarray, energy: float,
                         incident_phi=None) -> None:
        """_summary_

        Parameters
        ----------
        incident_wave : np.ndarray
            _description_
        energy : float
            _description_
        incident_phi : _type_, optional
            _description_, by default None

        Raises
        ------
        NotImplementedError
            _description_
        """
        raise NotImplementedError('Must overwrite transmit_fresnel')

    @abstractmethod
    def transmit_raytrace(self, incident_wave: np.ndarray,  energy: float,
                          incident_phi: np.ndarray) -> None:
        """_summary_

        Parameters
        ----------
        incident_wave : np.ndarray
            _description_
        energy : float
            _description_
        incident_phi : np.ndarray
            _description_

        Raises
        ------
        NotImplementedError
            _description_
        """
        raise NotImplementedError('Must overwrite transmit_raytrace')

    def show_sample(self) -> Figure:
        """_summary_

        Returns
        -------
        Figure
            _description_
        """
        fig, axs = plt.subplots(len(self.thickness), 1,
                                figsize=(6, len(self.thickness)*6))
        if not isinstance(axs, np.ndarray):
            axs = np.array([axs])
        fig.suptitle(f'{self.__class__.__name__}')
        for material, axis in zip(self.thickness, axs):
            plot = axis.imshow(self.thickness[material])
            axis.set_title(f'{material}')
            xticks = axis.get_xticks()
            yticks = axis.get_yticks()
            axis.xaxis.set_major_locator(mticker.FixedLocator(xticks))
            axis.yaxis.set_major_locator(mticker.FixedLocator(yticks))
            axis.set_xticklabels([f'{i*self.pixel_size:.1E}' for i in xticks])
            axis.set_yticklabels([f'{i*self.pixel_size:.1E}' for i in yticks])
            axis.tick_params(axis='x', rotation=45)
            fig.colorbar(plot, format='%.1e', ax=axis)
        fig.tight_layout()
        return fig

    def show_profile(self) -> Figure:
        """_summary_

        Returns
        -------
        Figure
            _description_
        """
        fig, axs = plt.subplots(len(self.thickness), 2,
                                figsize=(18, len(self.thickness)*6))
        fig.suptitle(f'{self.__class__.__name__}')
        try:
            axs_lst = np.array_split(axs.flatten(), axs.flatten().size//2)
        except AttributeError:
            axs_lst = np.array([axs])
        for material, axis in zip(self.thickness, axs_lst):
            axis[0].set_title(f'{material}: Horizontal Profile')
            axis[1].set_title(f'{material}: Vertical Profile')
            for i in np.unique(self.thickness[material], axis=0):
                axis[0].plot(i)
                xticks_0 = axis[0].get_xticks()
                axis[0].xaxis.set_major_locator(mticker.FixedLocator(xticks_0))
                axis[0].set_xticklabels(
                    [f'{i*self.pixel_size:.1e}' for i in xticks_0])
                # #   TODO fix this so that it does y in scientific notation
                # yticks_0 = axis[0].get_yticks()
                # print(list(yticks_0))
                # axis[0].yaxis.set_major_locator(mticker.FixedLocator(yticks_0))
                # axis[0].set_yticklabels([f'{i*self.pixel_size:.1e}' for i in yticks_0])
            axis[0].tick_params(axis='x', rotation=45)
            for j in np.unique(self.thickness[material].T, axis=0):
                axis[1].plot(j)
                xticks_1 = axis[1].get_xticks()
                axis[1].xaxis.set_major_locator(mticker.FixedLocator(xticks_1))
                axis[1].set_xticklabels(
                    [f'{i*self.pixel_size:.1e}' for i in xticks_1])
                axis[1].plot(j)
                # yticks_1 = axs[1].get_yticks()
                # axs[1].yaxis.set_major_locator(mticker.FixedLocator(yticks_1))
                # axs[1].set_yticklabels([f'{i*self.pixel_size:.1e}' for i in yticks_1])
            axis[1].tick_params(axis='x', rotation=45)
        fig.tight_layout()
        return fig

    # TODO weird stuff going on with horizontal gradient
    def show_gradient(self) -> Figure:
        """_summary_

        Returns
        -------
        Figure
            _description_
        """
        fig, axs = plt.subplots(len(self.gradient), 2,
                                figsize=(12, len(self.gradient)*6))
        fig.suptitle(f'{self.__class__.__name__}')
        try:
            axs_lst = np.array_split(axs.flatten(), axs.flatten().size//2)
        except AttributeError:
            axs_lst = np.array([axs])
        for material, axis in zip(self.gradient, axs_lst):
            axis[0].set_title(f'{material}: Vertical Gradient')
            axis[1].set_title(f'{material}: Horizontal Gradient')
            for i, grad in enumerate(self.gradient[material]):
                plot_grad = axis[i].imshow(grad)
                fig.colorbar(plot_grad, ax=axis[i], format='%.1e')
                xticks = axis[i].get_xticks()
                yticks = axis[i].get_yticks()
                axis[i].xaxis.set_major_locator(mticker.FixedLocator(xticks))
                axis[i].yaxis.set_major_locator(mticker.FixedLocator(yticks))
                axis[i].set_xticklabels(
                    [f'{j*self.pixel_size:.1e}' for j in xticks])
                axis[i].set_yticklabels(
                    [f'{j*self.pixel_size:.1e}' for j in yticks])
                axis[i].tick_params(axis='x', rotation=45)
        fig.tight_layout()

        return fig

    # TODO change ticklabels on figure
    def show_transmission(self, incident_wave: np.ndarray, energy: float,
                          incident_phi: np.ndarray = None) -> Figure:
        """_summary_

        Parameters
        ----------
        incident_wave : np.ndarray
            _description_
        energy : float
            _description_
        incident_phi : np.ndarray, optional
            _description_, by default None

        Returns
        -------
        Figure
            _description_
        """
        fig_f = None
        fig_rt = None

        try:
            tw_f = self.transmit_fresnel(incident_wave, energy, incident_phi)
            fig_f, axs = plt.subplots(2, 2, figsize=(15, 12))
            axs_lst = np.array_split(axs.flatten(), axs.flatten().size//2)
            fig_f.suptitle('Transmission')
            for wave, axis, title in zip((incident_wave, tw_f), axs_lst, ('Incident', 'Transmitted')):
                axis[0].set_title(f'{title}: Real')
                axis[1].set_title(f'{title}: Imaginary')
                plot_real = axis[0].imshow(np.real(wave))
                fig_f.colorbar(plot_real, ax=axis[0])
                plot_imag = axis[1].imshow(np.imag(wave))
                fig_f.colorbar(plot_imag, ax=axis[1])

            fig_f.tight_layout()
        except NotImplementedError:
            pass

        try:
            t_rt = self.transmit_raytrace(incident_wave, energy, incident_phi)
            fig_rt, axs = plt.subplots(
                len(self.materials), 2, figsize=(18, len(self.materials)*6))

            try:
                axs_lst = np.array_split(axs.flatten(), axs.flatten().size//2)
            except AttributeError:
                axs_lst = np.array([axs])

            for i, (image, axis) in enumerate(zip(t_rt, axs_lst)):
                plot_real = axis[0].imshow(np.real(image))
                fig_rt.colorbar(plot_real, ax=axs_lst[i][0])
                plot = axis[1].imshow(np.imag(image))
                fig_rt.colorbar(plot, ax=axs_lst[i][1])
                axis[0].set_title(f'{self.materials[i]}: Real')
                axis[1].set_title(f'{self.materials[i]}: Imaginary')
            fig_rt.tight_layout()

        except NotImplementedError:
            pass

        return fig_f, fig_rt

#   TODO check if translation is in pixels
    def _transform(self, image: np.ndarray) -> np.ndarray:
        """_summary_

        Parameters
        ----------
        image : np.ndarray
            _description_

        Returns
        -------
        np.ndarray
            _description_
        """
        if self.angle:
            rotated = rotate(image, self.angle)
        else:
            rotated = image
        distance = np.array(rotated.shape) - self.dimensions - 2*self.position
        translated = affine_transform(rotated,
                                      ((1, 0, distance[0]/2),
                                       (0, 1, distance[1]/2),
                                       (0, 0, 1)),
                                      output_shape=self.dimensions)
        # assert all(i == j for i, j in zip(self.dimensions, translated.shape))
        # ??? possibly multiply by pixel size
        # translated[translated < 0] = 0
        return translated


class AnalyticalSample(Sample):
    """


    Parameters
    ----------
    Sample : _type_
        _description_
    """
    _thickness = None

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def __add__(self, other: Type['AnalyticalSample']) -> Type['CompositeSample']:
        result = {k: self.thickness.get(k, 0) + other.thickness.get(k, 0)
                  for k in set(self.thickness) | set(other.thickness)}
        result = {k: v for k, v in result.items() if np.any(v)}
        if any(np.less(v, 0).any() for v in result.values()):
            warnings.warn('Thickness now contains negative values!')
        # if isinstance(self, Membrane2D) or isinstance(other, Membrane2D):
        #     return MultiLayerMembrane(result)
        return CompositeSample(result)

    # ??? can you have negative thickness?
    def __sub__(self, other: Type['AnalyticalSample']) -> Type['CompositeSample']:
        result = {k: self.thickness.get(k, 0) - other.thickness.get(k, 0)
                  for k in set(self.thickness) | set(other.thickness)}
        result = {k: v for k, v in result.items() if np.any(v)}
        if any(np.less(v, 0).any() for v in result.values()):
            warnings.warn('Thickness now contains negative values!')
        # if isinstance(self, Membrane2D) or isinstance(other, Membrane2D):
        #     return MultiLayerMembrane(result)
        return CompositeSample(result)

    # TODO Self in python 3.11
    def __mul__(self, other: Union[int, float]):
        result = {k: v*other for k, v in self.thickness.items()}
        if any(np.less(v, 0).any() for v in result.values()):
            warnings.warn('Thickness now contains negative values!')
        self.thickness = result
        return self

    def __rmul__(self, other: Union[int, float]):
        return self.__mul__(other)

    def __truediv__(self, other: Union[int, float]):
        result = {k: v/other for k, v in self.thickness.items()}
        if any(np.less(v, 0).any() for v in result.values()):
            warnings.warn('Thickness now contains negative values!')
        self.thickness = result
        return self

    @property
    def max_thickness(self) -> None:
        """_summary_

        Returns
        -------
        _type_
            _description_
        """
        return np.max(self.thickness)

    @property
    def gradient(self) -> np.ndarray:
        """_summary_

        Returns
        -------
        np.ndarray
            _description_
        """
        return {material: np.gradient(self.thickness[material],
                                      self.pixel_size, edge_order=2)
                                      for material in self.materials}

    def transmit_fresnel(self, incident_wave: np.ndarray, energy: float,
                         incident_phi: np.ndarray = None) -> np.ndarray:
        # TODO probably get rid of incident_phi
        # TODO np.prod
        refractive_index = self.refractive_index(energy)
        resulting_wave = incident_wave*np.exp(np.sum([1j*(refractive_index[i]-1)*self.thickness[i]*wavenumber(energy) for i in self.thickness], axis=0))
        return resulting_wave

    def transmit_raytrace(self, incident_wave: np.ndarray, energy: float, incident_phi: np.ndarray) -> np.ndarray:
        return super().transmit_raytrace(incident_wave, energy, incident_phi)


class CompositeSample(AnalyticalSample):
    """_summary_

    Parameters
    ----------
    Sample : _type_
        _description_
    """

    def __init__(self, thickness: Dict = None, **kwargs):
        super().__init__(**kwargs)
        if thickness is None:
            self._thickness = {None: np.zeros(self.dimensions)}
        else:
            self._thickness = thickness

    @property
    def thickness(self):
        return self._thickness  # [self._thickness < 0] == 0

    @thickness.setter
    def thickness(self, thickness):
        self._thickness = thickness

class VoxelizedSample(Sample):
    """_summary_

    Parameters
    ----------
    Sample : _type_
        _description_
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        raise NotImplementedError(
            'Voxelized samples are not currently implemented')


class Cylinder(AnalyticalSample):
    """


    Parameters
    ----------
    AnalyticalSample : _type_
        _description_
    """
    # _thickness = None

    def __init__(self, radius: float = 1.0, length: float = 1.0,
                 **kwargs) -> None:
        """_summary_

        Parameters
        ----------
        radius : float, optional
            _description_, by default 1.0
        length : float, optional
            _description_, by default 1.0
        """
        self.radius = radius
        self.length = length
        super().__init__(**kwargs)

    @property
    def thickness(self) -> np.ndarray:
        if self._thickness is not None:
            return self._thickness
        #   TODO non integer radius
        # radius = int(-((self.radius/self.pixel_size)//-1))  # Ceiling operator
        radius = round(self.radius/self.pixel_size)
        coordinates = np.arange(-radius, radius+1)
        cylinder = (radius/self.pixel_size)**2 - coordinates**2
        cylinder = np.add.outer(np.zeros(self.length), cylinder)
        cylinder[cylinder < 0] = 0
        cylinder = 2*np.sqrt(cylinder)
        cylinder = self._transform(cylinder)*self.pixel_size
        return {material: cylinder for material in self.materials}

    @thickness.setter
    def thickness(self, thickness):
        self._thickness = thickness


class SuperFormula(AnalyticalSample):
    """_summary_

    Parameters
    ----------
    AnalyticalSample : _type_
        _description_
    """

    def __init__(self):
        pass


# TODO sort out with different pixel sizes
# FIXME weird artifacts around it
class SuperQuadric(AnalyticalSample):
    """_summary_

    Parameters
    ----------
    AnalyticalSample : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    # _thickness = None

    def __init__(self, radii: Tuple = (1.0, 1.0, 1.0),
                 powers: Tuple = (2.0, 2.0, 2.0),
                 **kwargs) -> None:
        """_summary_

        Parameters
        ----------
        radii : Tuple, optional
            _description_, by default (1.0, 1.0, 1.0)
        powers : Tuple, optional
            _description_, by default (2.0, 2.0, 2.0)
        """
        # self._thickness = None
        if len(radii) == 3:
            self.radii = radii
        else:
            self.radii = tuple(list(radii)+((3-len(radii))*[1.0]))
        if len(powers) == 3:
            self.powers = powers
        else:
            self.powers = tuple(list(powers) + (3-len(powers))*[2.0])
        super().__init__(**kwargs)

    #TODO do check for <0 rather than isnan as faster
    @property
    def thickness(self):
        if self._thickness is not None:
            return self._thickness
        # x_coord = int(-((self.radii[2]/self.pixel_size)//-1))
        # y_coord = int(-((self.radii[1]/self.pixel_size)//-1))
        x_coord = round(self.radii[2]/self.pixel_size)
        y_coord = round(self.radii[1]/self.pixel_size)
        sq_x = np.abs(np.arange(-x_coord, x_coord)/x_coord)**self.powers[2]
        sq_y = np.abs(np.arange(-y_coord, y_coord)/y_coord)**self.powers[1]
        super_quadric = 1-np.add.outer(sq_y, sq_x)
        super_quadric[super_quadric < 0] = 0
        super_quadric **= 1/self.powers[0]
        super_quadric *= 2*self.radii[0]
        super_quadric[np.isnan(super_quadric)] = 0
        super_quadric = self._transform(super_quadric)
        super_quadric[super_quadric < 0] = 0
        return {material: super_quadric for material in self.materials}

    @thickness.setter
    def thickness(self, value) -> np.ndarray:
        self._thickness = value


class Sphere(SuperQuadric):
    """_summary_

    Parameters
    ----------
    SuperQuadric : _type_
        _description_
    """

    def __init__(self, radius: float = 1.0, **kwargs):
        """_summary_

        Parameters
        ----------
        radius : float, optional
            _description_, by default 1.0
        """
        xxx = [i for i in ('angle', 'powers', 'radii') if i in kwargs]
        if xxx:
            warnings.warn(
                f"""To use {' and '.join([', '.join(xxx[:-1]), xxx[-1]])
                if len(xxx) > 1 else xxx[0] if len(xxx) > 0 else ''}
                {'arg' if 'angle' in kwargs and len(xxx) == 1 else 'args'},
                use SuperQuadric instead."""
                .replace('\n', ' ').replace('  ', ''))
            for i in xxx:
                kwargs.pop(str(i))
        kwargs['radii'] = tuple(3*[radius])
        super().__init__(**kwargs)


class Ellipsoid(SuperQuadric):
    """_summary_

    Parameters
    ----------
    SuperQuadric : _type_
        _description_
    """

    def __init__(self, radii: Tuple = (1.0, 1.0, 1.0), **kwargs):
        """_summary_

        Parameters
        ----------
        radii : Tuple, optional
            _description_, by default (1.0, 1.0, 1.0)
        """
        kwargs['radii'] = radii
        if 'powers' in kwargs:
            kwargs.pop('powers')
            warnings.warn(
                'To use the powers argument, use SuperQuadric instead.')
        super().__init__(**kwargs)


# FIXME is really weird with changing pixel size
class Cone(AnalyticalSample):
    """_summary_

    Parameters
    ----------
    AnalyticalSample : _type_
        _description_
    """
    _thickness = None

    def __init__(self, radii: Union[float, Tuple] = (1.0, 1.0),
                 height: float = 1.0,
                 **kwargs):
        """_summary_

        Parameters
        ----------
        radii : Union[float, Tuple], optional
            _description_, by default (1.0, 1.0)
        height : float, optional
            _description_, by default 1.0
        """
        if isinstance(radii, (list, np.ndarray, tuple)):
            self.radius_y, self.radius_x = radii
        else:
            self.radius_y = self.radius_x = radii
        self.height = height
        super().__init__(**kwargs)

    @property
    def thickness(self) -> np.ndarray:
        if self._thickness is not None:
            return self._thickness
        radius_x = round(self.radius_x/self.pixel_size)
        radius_y = round(self.radius_y/self.pixel_size)
        cone_x = np.arange(-radius_x, radius_x+1)**2/radius_x**2
        cone_y = np.arange(-radius_y, radius_y+1)**2/radius_y**2
        cone = -np.sqrt(np.add.outer(cone_y, cone_x))
        cone += 1
        cone *= self.height
        # cone += self.height
        cone[cone < 0] = 0
        # cone -= self.height
        cone = self._transform(cone)  # ??? should it be to do with pixel_size
        return {material: cone for material in self.materials}

    @thickness.setter
    def thickness(self, value):
        self._thickness = value


class Torus(AnalyticalSample):
    """_summary_

    Parameters
    ----------
    AnalyticalSample : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    # _thickness = None

    def __init__(self, minor_radius: float = 1.0, major_radius: float = 1.0, **kwargs):
        self.minor_radius = minor_radius
        self.major_radius = major_radius
        super().__init__(**kwargs)

    @property
    def thickness(self):
        if self._thickness is not None:
            return self._thickness

        #   TODO make it elliptical
        minor_radius = round(self.minor_radius/self.pixel_size)
        major_radius = round(self.major_radius/self.pixel_size)
        coords = np.arange(-major_radius-minor_radius, major_radius+minor_radius+1)
        coords **= 2
        coords = np.sqrt(np.add.outer(coords, coords))
        coords -= self.major_radius/self.pixel_size
        # coords[coords < 0] = 0
        torus = (self.minor_radius/self.pixel_size)**2-coords**2
        torus[torus < 0] = 0
        torus = 2*np.sqrt(torus)
        torus = self._transform(torus)*self.pixel_size
        return {material: torus for material in self.materials}

    @thickness.setter
    def thickness(self, thickness):
        self._thickness = thickness

#   TODO: make a regular polygon class instead


class Cuboid(AnalyticalSample):
    """_summary_

    Parameters
    ----------
    AnalyticalSample : _type_
        _description_
    """
    # _thickness = None

    def __init__(self, sides: ArrayLike = (1, 1, 1), **kwargs):
        """_summary_

        Parameters
        ----------
        sides : ArrayLike, optional
            _description_, by default (1, 2, 2)
        """
        self.sides = sides
        super().__init__(**kwargs)

    @property
    def thickness(self):
        if self._thickness is not None:
            return self._thickness
        cuboid = self._transform(self.sides[0]*np.ones(self.sides[1:]))
        return {material: cuboid for material in self.materials}

    @thickness.setter
    def thickness(self, value):
        self._thickness = value


class Uniform(Cuboid):
    """_summary_

    Parameters
    ----------
    AnalyticalSample : _type_
        _description_
    """

    def __init__(self, thickness, **kwargs):
        xxx = [i for i in ('angle', 'position') if i in kwargs]
        if xxx:
            warnings.warn(
                f"""To use {' and '.join([', '.join(xxx[:-1]), xxx[-1]])
                if len(xxx) > 1 else xxx[0] if len(xxx) > 0 else ''}
                {'arg' if 'angle' in kwargs and len(xxx) == 1 else 'args'},
                use Cuboid instead."""
                .replace('\n', ' ').replace('  ', ''))
            for i in xxx:
                kwargs.pop(str(i))
        super().__init__(**kwargs)
        self.sides = (thickness, *self.dimensions)
