"""_summary_
"""
import numba as nb
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
from scipy.constants import c, e, h
# TODO use reikna to speed up fft
# TODO more propagation methods
# TODO AS, Direct Integration, FFT-DI


class Propagator(ABC):
    """_summary_
    """

    # TODO Sort out initialization
    def __init__(self, wavefront: np.ndarray, pixel_size: float) -> None:
        self.pixel_size = pixel_size
        self.wavefront = wavefront

    def __repr__(self) -> str:
        """
        String representation of the class

        Returns
        -------
        str
            The string representation of the class object
        """
        if self.__dict__.keys():
            i = max(map(len, list(self.__dict__.keys()))) + 1
            return '\n'.join([f'{k.rjust(i)}: {repr(v)}'
                              for k, v in sorted(self.__dict__.items())])
        return ''

    @abstractmethod
    def propagate(self, distances: np.ndarray, wavelength: float,
                  magnification: float) -> np.ndarray:
        """_summary_

        Parameters
        ----------
        distances : np.ndarray
            _description_
        wavelength : float
            _description_
        magnification : float
            _description_

        Returns
        -------
        np.ndarray
            _description_

        Raises
        ------
        NotImplementedError
            _description_
        """
        raise NotImplementedError('Must overwrite propagate')

    def show_propagation(self, distances: np.ndarray, wavelength: float,
                         magnification: float) -> Figure:
        """_summary_

        Parameters
        ----------
        distances : np.ndarray
            _description_
        wavelength : float
            _description_
        magnification : float
            _description_

        Returns
        -------
        Figure
            _description_
        """
        propagated = self.propagate(distances, wavelength, magnification)
        fig, axs = plt.subplots(
            propagated.shape[0], 2, figsize=(18, propagated.shape[0]*6))
        try:
            axs_lst = np.array_split(axs.flatten(), axs.flatten().size//2)
        except AttributeError:
            axs_lst = np.array([axs])
        for i, (image, axis) in enumerate(zip(propagated, axs_lst)):
            plot_real = axis[0].imshow(np.real(image))
            fig.colorbar(plot_real, ax=axs_lst[i][0])
            plot = axis[1].imshow(np.imag(image))
            fig.colorbar(plot, ax=axs_lst[i][1])
            axis[0].set_title(f'{distances[i]}: Real')
            axis[1].set_title(f'{distances[i]}: Imaginary')

        for _ in (axis.axis('off') for axes in axs_lst for axis in axes):
            pass
        fig.tight_layout()
        return fig


class Fresnel(Propagator):
    """_summary_

    Parameters
    ----------
    Propagator : _type_
        _description_
    """
    @property
    def wave_vector(self) -> np.ndarray:
        """_summary_

        Returns
        -------
        np.ndarray
            _description_
        """
        n_y, n_x = self.wavefront.shape
        k_x_coords = 2*np.pi*np.fft.fftfreq(n_x, self.pixel_size)
        k_y_coords = 2*np.pi*np.fft.fftfreq(n_y, self.pixel_size)
        k_x, k_y = np.meshgrid(k_x_coords, k_y_coords)
        return k_x**2+k_y**2

    def propagate(self, distances: np.ndarray, wavelength: float,
                  magnification: float) -> np.ndarray:
        """_summary_

        Parameters
        ----------
        distances : np.ndarray
            _description_
        wavelength : float
            _description_
        magnification : float
            _description_

        Returns
        -------
        np.ndarray
            _description_
        """
        #   TODO sort this bit out
        # if not isinstance(distances, (np.ndarray)):
        #     distances = np.array([distances])
        wavefront_f = np.fft.fft2(self.wavefront)
        propagated = np.zeros(
            [len(distances), *self.wavefront.shape], dtype=complex)
        for i, distance in enumerate(distances):
            if distance:
                exponent = -1j*distance*wavelength * \
                    self.wave_vector/(4*np.pi*magnification)
                propagated[i] = np.exp(1j*2*np.pi*distance /
                                       (wavelength*magnification)) * \
                    np.fft.ifft2(np.exp(exponent)*wavefront_f)
            else:
                propagated[i] = self.wavefront
        return propagated

    def propagate_laurene(self, distance, wavelength, magnification):
        """_summary_

        Parameters
        ----------
        distance : _type_
            _description_
        wavelength : _type_
            _description_
        magnification : _type_
            _description_
        """
        def getk(energy):
            """Get the wavenumber from energy in eV

            Parameters
            ----------
            energy : array_like
                _description_

            Returns
            -------
            _type_
                _description_
            """
            k = 2*np.pi*energy*e/(h*c)
            return k
        if not distance:
            return self.wavefront
        energy = h*c/(wavelength*e)
        k = getk(energy)

        Nx, Ny = self.wavefront.shape
        u, v = np.meshgrid(np.arange(0, Nx), np.arange(0, Ny))
        u = (u - (Nx / 2))
        v = (v - (Ny / 2))
        u_m = u * 2*np.pi / (Nx*self.pixel_size)
        v_m = v * 2*np.pi / (Ny*self.pixel_size)
        uv_sqr = np.transpose(u_m ** 2 + v_m ** 2)  # ie (u2+v2)

        waveAfterPropagation = np.exp(1j*k*distance/magnification)*np.fft.ifft2(np.fft.ifftshift(
            np.exp(-1j*distance*(uv_sqr)/(2*k*magnification))*np.fft.fftshift(np.fft.fft2(self.wavefront))))

        return waveAfterPropagation


# TODO raytraacing
class Raytrace(Propagator):
    """_summary_

    Parameters
    ----------
    Propagator : _type_
        _description_
    """

    def propagate(self, distances: np.ndarray, wavelength: float,
                  magnification: float) -> np.ndarray:
        return super().propagate(distances, wavelength, magnification)


# TODO try stacking them on 2nd axis and then do it (1st works 2nd probs faster)
# TODO test if padding and taking the centre region actually does anything
@nb.njit
def fastloopNumba(intensity_0, d_y, d_x, margin=0):
    """
    Accelerated part of the refraction calculation

    Args:
        Nx (int): shape[0] of the intensity refracted.
        Ny (int): shape[1] of the intensity refracted.
        intensityRefracted (2d numpy array): intensity before propag.
        intensityRefracted2 (2d numpy array): intensity after propag.
        Dy (2d numpy array): Displacement along x (in voxel).
        Dx (2d numpy array): Displacement along y (in voxel).

    Returns:
        intensityRefracted2 (2d numpy array): intensity after propag.

    """
    n_y, n_x = intensity_0.shape
    intensity_z = np.zeros((n_x+2*margin, n_y+2*margin), dtype=np.float64)
    for i, i_i in enumerate(intensity_0):
        for j, i_ij in enumerate(i_i):
            dx_ij = d_x[i, j]
            dy_ij = d_y[i, j]
            if not dx_ij and not dy_ij:
                intensity_z[i, j] += i_ij
                continue
            i_new = i
            j_new = j
            # Calculating displacement bigger than a pixel
            if np.abs(dx_ij) > 1:
                x = np.floor(dx_ij)
                i_new = np.int(i+x)
                dx_ij = dx_ij-x
            if np.abs(dy_ij) > 1:
                y = np.floor(dy_ij)
                j_new = np.int(j+y)
                dy_ij = dy_ij-y
            # Calculating sub-pixel displacement
            if 0 <= i_new < n_y and 0 <= j_new < n_x:
                intensity_z[i_new, j_new] += i_ij * \
                    (1-np.abs(dx_ij))*(1-np.abs(dy_ij))
                if i_new < n_y-1 and dx_ij >= 0:
                    if j_new < n_y-1 and dy_ij >= 0:
                        intensity_z[i_new+1, j_new] += i_ij*dx_ij*(1-dy_ij)
                        intensity_z[i_new+1, j_new+1] += i_ij*dx_ij*dy_ij
                        intensity_z[i_new, j_new+1] += i_ij*(1-dx_ij)*dy_ij
                    if j_new and dy_ij < 0:
                        intensity_z[i_new+1, j_new] += i_ij * \
                            dx_ij*(1-np.abs(dy_ij))
                        intensity_z[i_new+1, j_new-1] -= i_ij*dx_ij*dy_ij
                        intensity_z[i_new, j_new-1] += i_ij * \
                            (1-dx_ij)*np.abs(dy_ij)
                if i_new and dx_ij < 0:
                    if j_new < n_x-1 and dy_ij >= 0:
                        intensity_z[i_new-1, j_new] += i_ij * \
                            np.abs(dx_ij)*(1-dy_ij)
                        intensity_z[i_new-1, j_new+1] -= i_ij*dx_ij*dy_ij
                        intensity_z[i_new, j_new+1] += i_ij * \
                            (1-np.abs(dx_ij))*dy_ij
                    if j_new and dy_ij < 0:
                        intensity_z[i_new-1, j_new] += i_ij * \
                            np.abs(dx_ij)*(1-np.abs(dy_ij))
                        intensity_z[i_new-1, j_new-1] += i_ij*dx_ij*dy_ij
                        intensity_z[i_new, j_new-1] += i_ij * \
                            (1-np.abs(dx_ij))*np.abs(dy_ij)
    return intensity_z
