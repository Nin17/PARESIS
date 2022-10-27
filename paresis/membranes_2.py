"""_summary_
"""
import itertools
from abc import abstractmethod
from typing import Dict, Optional, Union, Type
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
from numpy.typing import ArrayLike
from scipy.ndimage import affine_transform
from scipy.constants import golden_ratio
from samples_2 import AnalyticalSample
import warnings
# TODO check that positions are within the FOV
#   TODO make positions and image show the same thing
# TODO make a completely random membrane where the shape
# TODO define a composite membrane class for checking if stuff is a sample or membrane
# TODO regular membrane with offset between rows 'keyboard membrane'
# TODO make positions show all materials and look nicer
# TODO make these all work properly with pixel size and the like
# kwargs are chosen randomly


class Membrane2D(AnalyticalSample):
    """_summary_

    Parameters
    ----------
    AnalyticalSample : _type_
        _description_
    ABC : _type_
        _description_

    Raises
    ------
    NotImplementedError
        _description_
    """

    def __init__(self, shape: Optional[AnalyticalSample] = None, shape_kwargs: Optional[Dict] = None, **kwargs):
        """_summary_

        Parameters
        ----------
        shape : AnalyticalSample
            _description_
        shape_kwargs : Dict
            _description_
        """
        self.shape = shape
        self.shape_kwargs = shape_kwargs
        super().__init__(**kwargs)

# TODO get the positions as well
    def __add__(self, other: Type['AnalyticalSample']) -> Type['CompositeSample']:
        result = {k: self.thickness.get(k, 0) + other.thickness.get(k, 0)
                  for k in set(self.thickness) | set(other.thickness)}
        result = {k: v for k, v in result.items() if np.any(v)}
        if any(np.less(v, 0).any() for v in result.values()):
            warnings.warn('Thickness now contains negative values!')
        # if isinstance(self, Membrane2D) or isinstance(other, Membrane2D):
        #     return MultiLayerMembrane(result)
        return MultiLayerMembrane(result)

    # ??? can you have negative thickness?
    def __sub__(self, other: Type['AnalyticalSample']) -> Type['CompositeSample']:
        result = {k: self.thickness.get(k, 0) - other.thickness.get(k, 0)
                  for k in set(self.thickness) | set(other.thickness)}
        result = {k: v for k, v in result.items() if np.any(v)}
        if any(np.less(v, 0).any() for v in result.values()):
            warnings.warn('Thickness now contains negative values!')
        # if isinstance(self, Membrane2D) or isinstance(other, Membrane2D):
        #     return MultiLayerMembrane(result)
        return MultiLayerMembrane(result)

    @property
    @abstractmethod
    def _positions(self):
        """_summary_

        Raises
        ------
        NotImplementedError
            _description_
        """
        raise NotImplementedError('Positions must be overwritten')

    #   TODO non-random orientations
    @property
    def thickness(self) -> np.ndarray:
        """_summary_

        Returns
        -------
        np.ndarray
            _description_
        """
        if self._thickness is not None:
            return self._thickness
        thickness = {material: 0 for material in self.materials}
        rng = np.random.default_rng()
        # TODO can definitely do this without looping over materials as well

        for i in self._positions:
            shape_thickness = self.shape(**self.shape_kwargs, dimensions=self.dimensions,
                                         position=np.array(i), angle=360*rng.uniform(),
                                         materials=(self.materials), pixel_size = self.pixel_size).thickness
            thickness = {k: thickness.get(k, 0) + shape_thickness.get(k, 0)
                         for k in thickness}

        return thickness

    @thickness.setter
    def thickness(self, value):
        self._thickness = value

    #   TODO sort this out
    def show_positions(self) -> Figure:
        """_summary_

        Returns
        -------
        _type_
            _description_
        """
        fig, axs = plt.subplots()
        axs.scatter(*-1*np.array(list(zip(*self._positions))))
        return fig

    #   TODO define translation and rotation methods possibly use np.roll
    #   TODO overwrite thickness method here rather than in all child classes


# TODO probably make it such that grains can't overlap
class UniformNoise():
    """A class for generating uniformly distributed, 2D noise."""

    def __init__(self, width: int = 0, height: int = 0, n: Union[int, None] = None) -> None:
        """Initialise the size of the domain and number of points to sample."""
        self.width, self.height = width, height
        if n is None:
            n = int(width * height)
        self.n = n

    def sample(self) -> np.ndarray:
        """_summary_

        Returns
        -------
        np.ndarray
            _description_
        """
        rng = np.random.default_rng()
        return zip(*[self.width*rng.random(self.n)-self.width//2,
                     self.height*rng.random(self.n)-self.height//2])


#   TODO probably inherit from composite sample instead
class WhiteMembrane(Membrane2D):
    """_summary_

    Parameters
    ----------
    Sample : _type_
        _description_
    """

    def __init__(self, n, **kwargs):
        self.n = n
        super().__init__(**kwargs)

    @property
    def _positions(self) -> zip:
        """_summary_

        Returns
        -------
        zip
            _description_
        """
        return UniformNoise(*self.dimensions, self.n).sample()


#   TODO make it so it's measured from the center
class PoissonDisc:
    """_summary_
    """

    def __init__(self, width: int, height: int, radius: int = 1, k: int = 30) -> None:

        self.width, self.height = width, height
        self.r = radius
        self.k = k
        # Cell side length
        self.a = radius/np.sqrt(2)
        # Number of cells in the x- and y-directions of the grid
        self.nx, self.ny = int(width / self.a) + 1, int(height / self.a) + 1

        self.cells = {coords: None for coords in itertools.product(
            range(self.nx), range(self.ny))}
        self.samples = []

    def _get_cell_coords(self, point: ArrayLike):
        """Get the coordinates of the cell that pt = (x,y) falls in."""

        return int(point[0] // self.a), int(point[1] // self.a)

    def _get_neighbours(self, coords: ArrayLike):
        """Return the indexes of points in cells neighbouring cell at coords.
        For the cell at coords = (x,y), return the indexes of points in the
        cells with neighbouring coordinates illustrated below: ie those cells
        that could contain points closer than r.

                                     ooo
                                    ooooo
                                    ooXoo
                                    ooooo
                                     ooo

        """
        for i, j in (k for k in itertools.product(range(-2, 3), repeat=2) if not (abs(k[0]) == 2 and abs(k[1]) == 2)):
            neighbour_coords = coords[0] + i, coords[1] + j
            if not (0 <= neighbour_coords[0] < self.nx and
                    0 <= neighbour_coords[1] < self.ny):
                # We're off the grid: no neighbours here.
                continue
            neighbour_cell = self.cells[neighbour_coords]
            if neighbour_cell is not None:
                # This cell is occupied: store the index of the contained point
                yield neighbour_cell

    def _point_valid(self, point: ArrayLike):
        """Is pt a valid point to emit as a sample?

        It must be no closer than r from any other point: check the cells in
        its immediate neighbourhood.

        """

        cell_coords = self._get_cell_coords(point)
        for idx in self._get_neighbours(cell_coords):
            nearby_pt = self.samples[idx]
            # Squared distance between candidate point, pt, and this nearby_pt.
            distance2 = (nearby_pt[0]-point[0])**2 + (nearby_pt[1]-point[1])**2
            if distance2 < self.r**2:
                # The points are too close, so pt is not a candidate.
                return False
        # All points tested: if we're here, pt is valid
        return True

    def _get_point(self, ref_point):
        """Try to find a candidate point near refpt to emit in the sample.

        We draw up to k points from the annulus of inner radius r, outer radius
        2r around the reference point, refpt. If none of them are suitable
        (because they're too close to existing points in the sample), return
        False. Otherwise, return the pt.

        """
        rng = np.random.default_rng()
        for i in itertools.count():
            # We failed to find a suitable point in the vicinity of refpt.
            if i > self.k:
                return False
            rho, theta = (self.r+2*self.r*rng.random(),
                          2*np.pi*rng.random())
            point = ref_point[0] + rho * \
                np.cos(theta), ref_point[1] + rho*np.sin(theta)
            if not (0 <= point[0] < self.width and 0 <= point[1] < self.height):
                # This point falls outside the domain, so try again.
                continue
            if self._point_valid(point):
                return point

    def sample(self):
        """
        Poisson disc random sampling in 2D.

        Draw random samples on the domain width x height such that no two
        samples are closer than r apart. The parameter k determines the
        maximum number of candidate points to be chosen around each reference
        point before removing it from the "active" list.

        Returns
        -------
        _type_
            _description_

        Yields
        ------
        Generator[tuple]
            _description_
        """

        rng = np.random.default_rng()
        # Pick a random point to start with.
        point = (self.width*rng.random(),
                 self.height*rng.random())
        self.samples = [point]
        # Our first sample is indexed at 0 in the samples list...
        self.cells[self._get_cell_coords(point)] = 0
        # and it is active, in the sense that we're going to look for more
        # points in its neighbourhood.
        active = [0]

        # As long as there are points in the active list, keep looking for
        # samples.
        while active:
            # choose a random "reference" point from the active list.
            idx = rng.choice(active)
            refpt = self.samples[idx]
            # Try to pick a new point relative to the reference point.
            point = self._get_point(refpt)
            if point:
                # Point pt is valid: add it to samples list and mark as active
                self.samples.append(point)
                nsamples = len(self.samples) - 1
                active.append(nsamples)
                self.cells[self._get_cell_coords(point)] = nsamples
            else:
                # We had to give up looking for valid points near refpt, so
                # remove it from the list of "active" points.
                active.remove(idx)
        return ((i-self.width//2, j-self.height//2) for i, j in self.samples)


class BlueMembrane(Membrane2D):
    """_summary_

    Parameters
    ----------
    Membrane2D : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """

    def __init__(self, radius, **kwargs):
        self.radius = radius
        super().__init__(**kwargs)

    @property
    def _positions(self):
        return PoissonDisc(*self.dimensions, self.radius).sample()


#   TODO generate enough points to cover FOV
class SpiralMembrane(Membrane2D):
    """_summary_

    Parameters
    ----------
    AnalyticalSample : _type_
        _description_
    """

    def __init__(self, a: float = 1.0, k: float = 20.0, n: float = 200.0,
                 **kwargs):
        self.a = a
        self.k = k
        self.n = n
        super().__init__(**kwargs)

    @property
    def _positions(self) -> zip:
        """_summary_

        Returns
        -------
        zip
            _description_
        """
        # FIXME number of points needs to be proportional to r
        kn = np.arange(np.pi/4, self.n*np.pi+np.pi/4)*self.k
        x = self.a*np.sqrt(kn)*np.cos(np.sqrt(kn))
        y = self.a*np.sqrt(kn)*np.sin(np.sqrt(kn))

        return zip(x, y)


# TODO generate enough points to fully cover FOV
class SunflowerMembrane(Membrane2D):
    """_summary_

    Parameters
    ----------
    AnalyticalSample : _type_
        _description_
    """
    # !!! Min radius = np.sqrt(a**2+b**2)/2

    def __init__(self, radius, **kwargs):
        self.radius = radius
        super().__init__(**kwargs)

    @property
    def _positions(self) -> zip:
        """_summary_

        Returns
        -------
        zip
            _description_
        """
        meh = np.arange(0, 720, 2)
        idk = self.radius*np.sqrt(meh)
        idk2 = (np.pi*meh/golden_ratio) % (2*np.pi)
        x = idk*np.cos(idk2)
        y = idk*np.sin(idk2)
        return zip(x, y)


class ConcentricMembrane(Membrane2D):
    """_summary_

    Parameters
    ----------
    Membrane2D : _type_
        _description_
    """

    def __init__(self, **kwargs) -> None:
        pass


#   TODO can definitely do this faster
class RegularMembrane(Membrane2D):
    """_summary_

    Parameters
    ----------
    Membrane2D : _type_
        _description_
    """

    def __init__(self, grid_spacing, **kwargs):
        self.grid_spacing = grid_spacing
        super().__init__(**kwargs)

    @property
    def _positions(self) -> zip:
        """_summary_

        Returns
        -------
        zip
            _description_
        """
        x_coords = np.linspace(-(self.dimensions[0]//2),
                               -(self.dimensions[0]//-2),
                               int(self.pixel_size*self.dimensions[0]/self.grid_spacing)+1)
        y_coords = np.linspace(-(self.dimensions[1]//2),
                               -(self.dimensions[1]//-2),
                               int(self.pixel_size*self.dimensions[1]/self.grid_spacing)+1)
        return itertools.product(x_coords, y_coords)

#   TODO tessalation of hexagons


# TODO get positions from other membranes
class MultiLayerMembrane(Membrane2D):
    """_summary_

    Parameters
    ----------
    Membrane2D : _type_
        _description_
    """

    def __init__(self, thickness: Optional[np.ndarray] = None, **kwargs):
        super().__init__(**kwargs)
        if thickness is None:
            self._thickness = {None: np.zeros(self.dimensions)}
        else:
            self._thickness = thickness

    @property
    def thickness(self):
        return self._thickness

    @property
    def _positions(self):
        raise NotImplementedError()
