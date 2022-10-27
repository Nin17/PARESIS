#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 14:37:59 2020

@author: quenot
"""
import datetime
import time
import os
import platform
import multiprocess as mp
from InputOutput.pagaille_io import save_edf
from Experiment import Experiment


def main(experiment_name, sample_sampling, nb_exp_points, margin,
         simulation_type, filepath, save=True, multiprocessing=True,
         cpus=None):
    """Main of the simulation code

    ----------
    experiment_name : str
        The name of the experiment in the .xml file
    sample_sampling : int
        Oversampling factor of the sample
    nb_exp_points : int
        Number of different membrane positions
    margin : int
        Prevent aliasing in Fresnel by extending images
    simulation_type : str
        The propagation method, must be Fresnel or RayT
    filepath : str
        The filepath to save the simulated images
    save : bool, optional
        Whether to save the output of the experiment in filepath, by default
        True
    multiprocessing : bool, optional
        Whether to process multiple energies in parallel, by default True
    cpus : int or None, optional
        The number of cores to use in the multiprocessing, by default None =
        All

    Raises
    ------
    ValueError
        If the given filepath is not found
    ValueError
        If simulation_type is neither Fresnel nor RayT
    """
    if all(i for i in multiprocessing.values()):
        raise ValueError('''Can only multiprocess for positions or energies
        independently''')
        # FIXME (with both multiprocessing as True)
        # AssertionError: daemonic processes are not allowed to have children
    if platform.system() == 'Windows' and \
            any(i for i in multiprocessing.values()):
        raise OSError('Multiprocessing not implemented in windows')
    exp_dict = {}
    time0 = time.time()  # timer for computation
    # ************************************************************************
    now = datetime.datetime.now()
    exp_dict['expID'] = now.strftime("%Y%m%d-%H%M%S")  # define experiment ID

    # Define experiment
    exp_dict['experimentName'] = experiment_name
    exp_dict['sampleSampling'] = sample_sampling
    exp_dict['nbExpPoints'] = nb_exp_points
    exp_dict['margin'] = margin
    exp_dict['simulation_type'] = simulation_type
    exp_dict['Multiprocessing'] = multiprocessing
    exp_dict['CPUs'] = cpus
    exp_images_filepath = filepath + \
        f"{exp_dict['simulation_type']}{exp_dict['expID']}/"
    exp_dict['filepath'] = exp_images_filepath
    experiment = Experiment(exp_dict)

    if save and not os.path.isdir(filepath):
        raise ValueError('Path not found')

    if save:
        if not os.path.isdir(exp_images_filepath):
            os.mkdir(exp_images_filepath)
        if not os.path.isdir(exp_images_filepath+'membrane/'):
            os.mkdir(exp_images_filepath+'membrane/')
        thresholds = experiment.myDetector.myBinsThresholds.copy()
        if thresholds:
            thresholds.insert(0, experiment.mySource.mySpectrum[0][0])
        else:
            thresholds = [experiment.mySource.mySpectrum[0][0]]*2
        nbin = len(thresholds) - 1
        exp_path_en = []
        for i in range(nbin):
            binstart = f'{int(thresholds[i]):02d}'
            binend = f'{int(thresholds[i+1]):02d}'
            exp_path_en.append(f'{exp_images_filepath}{binstart}_\
                {binend}kev/')
            if len(thresholds)-1 == 1:
                exp_path_en = [exp_images_filepath]
            if save:
                dirs = ['white', 'reference/', 'sample/', 'propagation/']
                for folder in dirs:
                    if not os.path.isdir(exp_path_en[i]+folder):
                        os.mkdir(exp_path_en[i]+folder)

    if exp_dict['simulation_type'] not in ('Fresnel', 'RayT'):
        raise ValueError(f"Simulation Type ({exp_dict['simulation_type']})\
            must be either Fresnel or RayT")

    propagators = {'Fresnel': experiment.computeSampleAndReferenceImages,
                   'RayT': experiment.computeSampleAndReferenceImagesRT}

    def _parallel_positions(point_num):
        experiment.myMembrane.myGeometry = []
        experiment.myMembrane.getMyGeometry(
            experiment.studyDimensions,
            experiment.myMembrane.membranePixelSize,
            experiment.sampling, point_num,
            exp_dict['nbExpPoints'])
        print("\n\nINITIALIZING EXPERIMENT PARAMETERS AND GEOMETRIES")
        print("\n\n*************************")
        print("Calculations point", point_num)
        sample_image_tmp, reference_image_tmp, propag_image_tmp, white = \
            propagators[exp_dict['simulation_type']](point_num)

        if save:
            save_edf(experiment.myMembrane.myGeometry[0], exp_images_filepath +
                     f"membrane/{exp_dict['experimentName']}_membrane" +
                     f"{exp_dict['sampleSampling']}_{point_num:02d}.edf")
            for ibin in range(nbin):
                save_edf(sample_image_tmp[ibin], exp_path_en[ibin] +
                         f"sample/sample_{exp_dict['expID']}_" +
                         f"{point_num:02d}.edf")
                save_edf(reference_image_tmp[ibin], exp_path_en[ibin] +
                         f"reference/reference_{exp_dict['expID']}_" +
                         f"{point_num:02d}.edf")
                if point_num == 0:
                    save_edf(propag_image_tmp[ibin], exp_path_en[ibin] +
                             "propagation/propagation_" +
                             f"{exp_dict['expID']}.edf")
                    save_edf(white[ibin], exp_path_en[ibin] +
                             f"white/white_{exp_dict['expID']}" +
                             '_.edf')

    # if multiprocessing['positions']:
    #     with mp.Pool(processes=4) as pool:  # TODO choose a better number
    #         pool.map(_parallel_positions, range(exp_dict['nbExpPoints']))
    #         pool.close()
    #         pool.join()
    # else:
    for i in range(exp_dict['nbExpPoints']):
        _parallel_positions(i)

    if save:
        experiment.saveAllParameters(time0, exp_dict)
    print("\nfini")


if __name__ == '__main__': 
    main("Energies6", sample_sampling=3, nb_exp_points=5, margin=10,
         simulation_type='RayT', filepath='/Users/chris/Desktop/DATA/',
         save=True, multiprocessing={'positions': False, 'energies': False})
