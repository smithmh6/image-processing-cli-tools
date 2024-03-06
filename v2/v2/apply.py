"""
This module contains the apply_model() function used when
applying an existing model to a dataset.
"""

# import dependencies
import ast
from configparser import ConfigParser
import cv2 as cv
import h5py
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
from uster_processing.utils import *
from uster_processing.shapes import Point, Rectangle
import sys
from scipy.optimize import curve_fit
from scipy.signal import convolve2d


def apply_model(cfg:ConfigParser):
    """
    Applies an existing model to a new dataset.
    """

    # open the data file (hdf5 file)
    logging.info(
        f"Reading {os.path.getsize(cfg['model']['model_path'])} bytes from model {cfg['model']['model_path']}.")
    # read the existing model file in readonly mode
    model = h5py.File(cfg['model']['model_path'], 'r')

    # store the images
    imgs = get_images(cfg)


    # calculate approximate memory usage of images
    img_mem = np.sum([sys.getsizeof(v) for v in imgs.values()])
    logging.info(f"Allocated {img_mem} bytes for image data.")

    # get the plot variables
    plot_vars = ast.literal_eval(cfg['output']['plot_vars'])
    logging.info(f"Found plot variables ---> {plot_vars}.")

    # plot input images, if requested
    if 'inputs' in plot_vars:
        plot_images(
            imgs,
            cfg['paths']['out'],
            cfg['paths']['run_label'],
            'inputs',
            cfg.getboolean('output', 'show_plots'),
            cfg['output']['cmap']
        )

    # flatfield images if requested
    if cfg.getboolean('images', 'flatfield'):
        ff_region = ast.literal_eval(cfg['images']['ff_region'])
        imgs = flatfield(ff_region, imgs)

        if 'flatfield' in plot_vars:
            plot_images(
                imgs,
                cfg['paths']['out'],
                cfg['paths']['run_label'],
                'flatfields',
                cfg.getboolean('output', 'show_plots'),
                cfg['output']['cmap']
            )

    # calculate ND ratios if requested
    if cfg['model']['mode'].lower() == 'ratio':
        imgs = ratiometric(imgs)

        if 'ratiometric' in plot_vars:
            plot_images(
                imgs,
                cfg['paths']['out'],
                cfg['paths']['run_label'],
                'ratiometric',
                cfg.getboolean('output', 'show_plots'),
                cfg['output']['cmap']
            )


    # extract fit function from model
    logging.info("Parsing model data.")
    logging.info(f"Found fit function --> {int(model['model'].attrs['fit_order'])}.")
    fit_fx = fit_func(int(model['model'].attrs['fit_order']), len(imgs))

    # polynomial coefficients
    opt = np.asarray(model['model']['opt'])
    logging.info(f"Found coefficients --> {opt}.")

    # roc curve data
    roc = {
        't_val': model['roc']['t_val'][()],
        'AUROC': model['roc']['AUROC'][()]
    }
    logging.info(f"Found AUROC= {roc['AUROC']} and threshold= {roc['t_val']}.")

    # create a dict to save output images
    out_imgs = {}

    # generate the score image
    logging.info("Generating score image.")
    score_fx = fit_fx(np.array([v.flatten() for v in imgs.values()]), *opt)
    score_img = np.reshape(score_fx, list(imgs.values())[0].shape)
    out_imgs['Score'] = score_img

    # perform smoothing
    if cfg.getint('images', 'smooth') > 0:
        logging.info(f"Performing smoothing with kernel size {cfg['images']['smooth']}.")
        k_smooth = create_kernel(cfg.getint('images', 'smooth'), 'smooth')
        smooth_img = convolve2d(score_img, k_smooth, mode='same')
        out_imgs['Smooth'] = smooth_img

    # perform thresholding
    logging.info(f"Generating detection image with ROC threshold --> {roc['t_val']}.")
    # if smoothing requested, use smoothed image for detection
    if cfg.getint('images', 'smooth') > 0:
        _, detect_img = cv.threshold(smooth_img, roc['t_val'], 1, cv.THRESH_BINARY)
    else:
        _, detect_img = cv.threshold(score_img, roc['t_val'], 1, cv.THRESH_BINARY)

    # add detection image to output dict
    out_imgs['Detection'] = detect_img


    # apply erosion if requested
    if cfg.getint('images', 'erosion') > 0:
        logging.info(f"Performing erosion with kernel size {cfg['images']['erosion']}.")
        k_erode = create_kernel(cfg.getint('images', 'erosion'), 'erosion')
        erode_img = cv.erode(detect_img, k_erode)
        out_imgs['Eroded'] = erode_img

    # apply dilation if requested
    if cfg.getint('images', 'dilation') > 0:
        logging.info(f"Performing dilation with kernel size {cfg['images']['dilation']}.")
        k_dilate = create_kernel(cfg.getint('images', 'dilation'), 'dilation')

        # if erosion was requested, use eroded image
        if cfg.getint('images', 'erosion') > 0:
            dilate_img = cv.dilate(erode_img, k_dilate)
        else:
            dilate_img = cv.dilate(detect_img, k_dilate)

        # add dilated image to output image dict
        out_imgs['Dilated'] = dilate_img

    # apply false coloring if requested
    if cfg.getboolean('output', 'overlay'):
        logging.info("Generating color overlay image.")

        # last item in dict is classification img
        class_img = list(out_imgs.values())[len(out_imgs) - 1]
        overlay_img = color_overlay(list(imgs.values())[0], class_img.astype(np.float32))
        # add to output dict
        out_imgs['Overlay'] = overlay_img


    # close the output file
    model.close()

    plot_images(
        out_imgs,
        cfg['paths']['out'],
        cfg['paths']['run_label'],
        'output',
        cfg.getboolean('output', 'show_plots'),
        cfg['output']['cmap'],
        blocking=True
    )

    plt.close('all')

