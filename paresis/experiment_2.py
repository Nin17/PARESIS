"""Create experiments
"""
import json
import os
import warnings
from typing import Dict, Tuple
import numpy as np
from numpy.typing import ArrayLike
from source_2 import Source
from detector_2 import Detector
from propagator_2 import Fresnel, Propagator
from samples_2 import Cylinder, Sphere, Sample, CompositeSample
from membranes_2 import Membrane2D, WhiteMembrane, MultiLayerMembrane
from utilities import Utility
# TODO attenuation in surrounding material before samples and in region around samples
# TODO actual phase shift and attenuation
# TODO to_json probably will need to overwrite this
# tODO check membrane and sample dimensions better
# tODO could use quantities module
# TODO samples and membranes possibly at multiple distnaces, 2d tuple
#TODO geometry class


class Geometry:
    """_summary_
    """
    def __init__(self, source_membrane: ArrayLike = None,
                 source_sample: ArrayLike = None,
                 sample_detector: ArrayLike = None) -> None:
        """_summary_

        Parameters
        ----------
        source_membrane : ArrayLike, optional
            _description_, by default None
        source_sample : ArrayLike, optional
            _description_, by default None
        sample_detector : ArrayLike, optional
            _description_, by default None
        """
        self._source_membrane = source_membrane
        self._source_sample = source_sample
        self._sample_detector = sample_detector
        # TODO calculate magnification and that malarky

    @property
    def source_membrane(self) -> ArrayLike:
        """_summary_

        Returns
        -------
        _type_
            _description_
        """
        return self._source_membrane

    @property
    def source_sample(self) -> ArrayLike:
        """_summary_

        Returns
        -------
        _type_
            _description_
        """
        return self._source_sample

    @property
    def sample_detector(self) -> ArrayLike:
        """_summary_

        Returns
        -------
        _type_
            _description_
        """
        return self._sample_detector

    @property
    def magnification(self):
        return


class Experiment(Utility):
    """_summary_
    """

    def __init__(self, distances: Dict = None, name: str = '', filepath: str = '/Users/chris/Desktop',
                 sources: Tuple = None,
                 detector: Detector = None,
                 samples: Tuple = None,
                 membranes: Tuple = None,
                 propagator: Propagator = Fresnel, save: bool = False,
                 json_file: str = 'json_files/experiment.json') -> None:
        """_summary_

        Parameters
        ----------
        source : Source, optional
            _description_, by default Source()
        detector : Detector, optional
            _description_, by default Detector()
        samples : Tuple, optional
            _description_, by default (Cylinder(), Sphere())
        propagator : Propagator, optional
            _description_, by default Fresnel
        """
        # TODO probably do this check after loading stuff from file ...
        if membranes is None:
            membranes = ()
        if samples is None:
            samples = ()
        dimensions_set = set((i.dimensions for i in samples+membranes))
        if len(dimensions_set) != 1:
            raise ValueError(f'All sample and membrane dimensions must be the same, currently they include {dimensions_set}')
        self.initial_wave = np.ones((list(dimensions_set)[0]))
        pixel_size_set = set((i.pixel_size for i in samples+membranes))
        if len(pixel_size_set) != 1:
            raise ValueError(f'All sample and membrane pixel sizes must be the same, currently they include {pixel_size_set}')
        
        # TODO an oversampling one as well at some point
        # membrane_dimensions = set((i.dimensions for ))
        super().__init__(name, json_file)
        self.distances = distances
        self.filepath = filepath
        if self.name not in self.filepath:
            self.filepath = os.path.join(
                self.filepath, self.name)
        self.sources = sources
        self.detector = detector
        if any(isinstance(i, Membrane2D) for i in samples):
            warnings.warn('There is a membrane in the samples list')
        self.sample = sum((i for i in samples), CompositeSample()) if len(samples) >1 else samples[0]
        self.sample.show_gradient()
        if not all(isinstance(i, Membrane2D) for i in membranes):
            warnings.warn('There is a non-membrane object in the membranes')
        self.membrane = sum((i for i in membranes), MultiLayerMembrane()) if len(membranes) > 1 else membranes[0]
        self.membrane.show_gradient()
        self.propagator = propagator

        if save:
            self._mk_dir()

    def _mk_dir(self) -> None:
        """_summary_
        """
        if not os.path.isdir(self.experiment['filepath']):
            os.mkdir(self.experiment['filepath'])
        dirs = ['absorption', 'phase', 'propagation',
                'reference', 'sample', 'white']
        for source in self.sources:
            source_path = os.path.join(
                self.experiment['filepath'], source.name)
            if not os.path.isdir(source_path):
                os.mkdir(source_path)
            for i in dirs:
                path = os.path.join(source_path, i)
                if not os.path.isdir(path):
                    os.mkdir(path)

    def _rm_dir(self) -> None:
        """_summary_
        """
        for (dirpath, _, _) in reversed(list(
                os.walk(self.experiment['filepath']))):

            try:
                os.rmdir(dirpath)
            except OSError:
                pass

    # TODO check that this actually works
    @property
    def binned_spectra(self):
        """_summary_
        """
        print('woop')
        splits = [np.searchsorted(i.spectrum[0], self.detector.bin_thresholds, side='right') for i in self.sources]
        binned_spectra = [np.split(self.sources[i].spectrum, split) for i, split in enumerate(splits)]
        return binned_spectra

    def sample_images(self) -> np.ndarray:
        """_summary_

        Returns
        -------
        np.ndarray
            _description_
        """

        return

    def reference_images(self) -> np.ndarray:
        """_summary_

        Returns
        -------
        np.ndarray
            _description_
        """
        idk = [i for i in self.samples if isinstance(i, Membrane2D)]
        return

    def white_field(self) -> np.ndarray:
        """_summary_

        Returns
        -------
        np.ndarray
            _description_
        """
        return

    #TODO make a generator for the propagation images
    def _propagation_images(self, n: int = 1):
        yield 1

    def propagation_images(self, n: int = 1) -> np.ndarray:
        """_summary_

        Returns
        -------
        np.ndarray
            _description_
        """
        idk = [i.thickness for i in self.samples if not isinstance(
            i, Membrane2D)]
        thickness = np.sum(idk, axis=0)
        propagated = self.propagator(
            1-thickness/np.max(thickness), pixel_size=1e-6).propagate([20], 1e-11, 1)
        return self.detector.detect(np.real(propagated[0])+np.finfo(float).eps, 10)
        # return np.real(propagated)

    # ??? Maybe this saves too much info
    # TODO the source saving bit better
    def save_parameters(self) -> None:
        """_summary_
        """
        filepath = os.path.join(
            self.filepath,
            f"{self.name}_parameters.txt")
        with open(filepath, 'w', encoding='UTF-8') as file:
            for i in sorted(self.__dict__):
                file.write(f'{i}:\n\n'.upper())
                attr = getattr(self, i)
                if isinstance(attr, dict):
                    j = max(map(len, list(attr.keys()))) + 1
                    file.write('\n'.join([f'{k.rjust(j)}: {repr(v)}'
                                          for k, v in sorted(attr.items())]))
                elif not isinstance(attr, type):
                    if isinstance(attr, (list, tuple)):
                        for k in attr:
                            if isinstance(k, Sample):
                                file.write(f'{k}')
                            else:
                                file.write(f'{k}\n\n')
                    else:
                        file.write(f'{attr}')

                else:
                    file.write(f'\t{attr.__name__}')
                if i != sorted(self.__dict__)[-1]:
                    file.write('\n\n')
            file.close()
