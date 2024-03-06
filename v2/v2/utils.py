"""
This module contains helper functions used in
USTER data processing.
"""

# import dependencies
import ast
from configparser import ConfigParser
import cv2 as cv
import logging
import matplotlib.pyplot as plt
import h5py
from math import sqrt
import numpy as np
import os


def get_images(cfg:ConfigParser) -> dict:
    """
    Extracts images from data file and
    returns a dictionary.
    """

    # store the images
    imgs = {}

    # open the data file (hdf5 file)
    logging.info(
        f"Reading {os.path.getsize(cfg['paths']['data'])} bytes from {cfg['paths']['data']}.")
    with h5py.File(cfg['paths']['data']) as df:

        # extract the image dataset --> data/dataset/images
        logging.info(f"Extracting dataset {cfg['data']['dataset']}.")
        ds = df['data'][cfg['data']['dataset']][cfg['data']['run']]['images']


        # get the moe names from config file
        filters = ast.literal_eval(cfg['data']['filters'])
        sf = float(cfg['data']['scale_factor'])
        logging.info(f"Found {len(filters)} filters --> {filters}.")
        logging.info(f"Loading images with scaling --> {sf}.")

        logging.info(f"Extracting and scaling numerical image data.")
        sf = float(cfg['data']['scale_factor'])
        for k, v in list(ds.items()):
            fil_name = str(k).split('.')[0]
            if fil_name in filters:
                # extract image array and scale 16-bit int to float
                imgs[fil_name] = np.asarray(v).astype(np.float64) / sf

    return imgs

def plot_images(
    imgs:dict, out:str, run_label:str, name:str,
    show:bool, cmap:str, blocking:bool=False) -> None:
    """
    Plots the input images.
    """

    fig = plt.figure(num=name, figsize=(10, 8))

    if len(imgs) < 4:
        fig_rows = 1
        fig_cols = len(imgs)
    if len(imgs) == 4:
        fig_rows = 2
        fig_cols = 2
    elif 4 < len(imgs) <= 6:
        fig_rows = 2
        fig_cols = 3
    elif 6 < len(imgs) <= 8:
        fig_rows = 4
        fig_cols = 4

    logging.info(f"Constructing plot {run_label}_{name}.png.")
    for i in range(len(imgs)):
        plt.subplot(fig_rows, fig_cols, i + 1)
        plt.title(list(imgs.keys())[i])
        plt.imshow(list(imgs.values())[i], cmap=cmap)
    plt.tight_layout()

    # display gui if requested
    if show == True:
        if blocking:
            plt.show()
        else:
            plt.show(block=blocking)
            plt.pause(0.1)

    # save the figure
    fig.savefig(os.path.join(out, run_label, f'{run_label}_{name}.png'))

    # close the resource
    #plt.close()


def flatfield(rows:tuple, imgs:dict) -> dict:
    """
    Performs flatfielding using a user selected
    sections of rows in the images. Returns a copy of
    the input images after flatfielding.
    """

    logging.info(f"Performing flatfielding with pixel range --> {rows}.")

    # deep copy input images
    ff_imgs = {k: v for k, v in imgs.items()}

    for k, im in ff_imgs.items():

        # get flatfield region average
        ff_reg = np.mean(im[rows[0]:rows[1], :], axis=0)
        temp = im / ff_reg

        # overwrite the values
        ff_imgs[k] = temp

    return ff_imgs

def ratiometric(imgs:dict) -> dict:
    """
    Ratio the images with the ND filter.
    Returns a copy of input images after
    ratio calculations.
    """

    # deep copy input images
    ratio_imgs = {k: v for k, v in imgs.items()}

    # get the ND image
    for k, v in ratio_imgs.items():
        if not 'MOE' in k:
            ndf_key = k
            ndf_arr = v
            break

    # remove the ndf image from dict
    ratio_imgs.pop(ndf_key)

    logging.info(f"Performing ratiometric calculations with filter {k}.")

    # divide MOE images by ND filter
    for k, v in ratio_imgs.items():
        # perform ratio calculation
        temp = v / ndf_arr
        # only overwrite MOE images
        ratio_imgs[k] = temp

    return ratio_imgs

def extract_roi(imgs:dict, roi:list) -> np.ndarray:
    """
    Extracts pixel intensities from list of ROI regions.
    """

    # store roi's
    rois = []
    for im in imgs.values():
        roi_arr = np.array([])

        for r in roi:
            # slice the image array
            r = np.asarray(
                im[r.top_left.y : r.bottom_left.y + 1, r.top_left.x : r.top_right.x + 1])
            # combine with previous target array
            roi_arr = np.concatenate((roi_arr, r.flatten()))

        rois.append(roi_arr)

    return np.asarray(rois)

def fit_func(order:int, vars:int):
    """
    Returns the fit function to be used
    for the MOE model.
    """
    if order == 1:
        if vars == 1:
            return func_11
        elif vars == 2:
            return func_12
        elif vars == 3:
            return func_13
        elif vars == 4:
            return func_14
    elif order == 2:
        if vars == 1:
            return func_21
        elif vars == 2:
            return func_22
        elif vars == 3:
            return func_23
        elif vars == 4:
            return func_24
    else:
        raise ValueError(f"Function order {order} with {vars} filters not found.")

def func_11(data, a, b):
    """"""
    return (a * data[0]) + b

def func_12(data, a, b, c):
    """"""
    return (a * data[0]) + (b * data[1]) + c

def func_13(data, a, b, c, d):
    """"""
    return (a * data[0]) + (b * data[1]) + (c * data[2]) + d

def func_14(data, a, b, c, d, e):
    """"""
    return (a * data[0]) + (b * data[1]) + (c * data[2]) + (d * data[3]) + e

def func_21(data, a, b, c):
    """"""
    return (
        (a * data[0]**2) + (b * data[0]) + c
    )

def func_22(data, a, b, c, d, e):
    """"""
    return (
        (a * data[0]**2) + (b * data[0]) +
        (c * data[1]**2) + (d * data[1]) + e
    )

def func_23(data, a, b, c, d, e, f, g):
    """"""
    return (
        (a * data[0]**2) + (b * data[0]) +
        (c * data[1]**2) + (d * data[1]) +
        (e * data[2]**2) + (f * data[2]) + g
    )

def func_24(data, a, b, c, d, e, f, g, h, i):
    """"""
    return (
        (a * data[0]**2) + (b * data[0]) +
        (c * data[1]**2) + (d * data[1]) +
        (e * data[2]**2) + (f * data[2]) +
        (g * data[3]**2) + (h * data[3]) + i
    )


def roc_curve(truth:np.ndarray, detections:np.ndarray, thresh:np.ndarray) -> dict:
    """
    Calculate the Receiver Operator Characteristic
    (ROC) curve to find the optimal threshold.

    Parameters
    -----------
    truth (np.ndarray): logical index or binary values indicating class membership
            detections - measured results (or prediction scores) from sensor\n
    detections (np.ndarray): measured results (or prediction scores) from sensor\n
    thresh (np.ndarray): user specified threshold values for computing the ROC curve

    Returns
    ---------
    dictionary object of ROC values\n
    {key : value}\n
    { 'AUROC' : Area Under the Receiver Operator Curve,\n
    'Pd' : Probability of detection (or sensitivity),\n
    'Pfa' : Probability of false alarm (or 1 - specificity),\n
    't_ind' : index of optimal threshold,\n
    't_val' : Optimal threshold based on distance to origin,\n
    'Se' : Optimal sensitivity based upon optimal threshold,\n
    'Sp' : Optimal specificity based upon optimal threshold }
    """

    # validate input data
    if not truth.shape == detections.shape:
        raise ValueError(
            f'Shape mismatch at positions 1 and 2. Received {truth.shape} and {detections.shape}')
    if not isinstance(truth, np.ndarray):
        raise TypeError(f'truth expected <np.ndarray> but received {type(truth)}.')
    if not isinstance(detections, np.ndarray):
        raise TypeError(f'detections expected <np.ndarray> but received {type(detections)}.')
    if not isinstance(thresh, np.ndarray):
        raise TypeError(f'thresh expected <np.ndarray> but received {type(thresh)}.')


    # define a confusion matrix as a dict
    roc_matrix = {
        'true_pos':np.zeros(len(thresh)),
        'true_neg':np.zeros(len(thresh)),
        'false_pos':np.zeros(len(thresh)),
        'false_neg':np.zeros(len(thresh)),
        'prob_det':np.zeros(len(thresh)),
        'prob_fa':np.zeros(len(thresh))
    }

    # Run loop to threshold detections data and calculate TP, TN, FP & FN
    for i, val in enumerate(thresh):

        temp_detects = np.asarray([1 if d >= val else 0 for d in detections])
        tp_temp = np.sum(truth * temp_detects)

        roc_matrix['true_neg'][i] = np.sum((1 - truth) * (1 - temp_detects))
        roc_matrix['false_pos'][i] = np.sum(temp_detects) - tp_temp
        roc_matrix['false_neg'][i] = np.sum(truth) - tp_temp
        roc_matrix['true_pos'][i] = tp_temp

    # Calculate Pd and Pfa
    roc_matrix['prob_det'] = (
        roc_matrix['true_pos'] / (roc_matrix['true_pos'] + roc_matrix['false_neg']))
    roc_matrix['prob_fa'] = (
        1 - (roc_matrix['true_neg'] / (roc_matrix['false_pos'] + roc_matrix['true_neg'])))

    # map points to distance from upper left corner (0, 1)
    euc_dist = list(
        map(lambda x: sqrt((0 - x[0])**2 + (1 - x[1])**2),
            zip(roc_matrix['prob_fa'], roc_matrix['prob_det'])))

    # Find the best threshold index
    t_ind = np.argmin(euc_dist)

    # Calculate the AUROC using a simple summation of rectangles
    au_roc = -np.trapz(
        np.append(roc_matrix['prob_det'], 0), np.insert(roc_matrix['prob_fa'], 0, 1))

    return {
        'AUROC':au_roc,                         # area under roc curve
        'Pd':roc_matrix['prob_det'],            # probability of detection
        'Pfa':roc_matrix['prob_fa'],            # probability of false alarm
        't_val':thresh[t_ind],                  # roc threshold value
        'Se':roc_matrix['prob_det'][t_ind],     # sensitivity
        'Sp':1 - roc_matrix['prob_fa'][t_ind]   # specificity
    }

def create_kernel(k:int, k_type:str) -> np.ndarray:
    """
    Creates a kernel array based on k_type.
    """

    if k_type == 'smooth':
        return np.ones((k, k), np.float32) / k**2

    else:
        return np.ones((k, k), np.uint8)


def color_overlay(raw_img:np.ndarray, class_img:np.ndarray) -> np.ndarray:
    """
    Overlays raw image with class image.
    """

    # scale images to float 32
    raw_img = cv.normalize(
        raw_img.astype(np.float32),
        dst=None,
        alpha=0,
        beta=1,
        norm_type=cv.NORM_MINMAX
    )
    ## class_img = cv.normalize(
    ##     class_img.astype(np.float32),
    ##     dst=None,
    ##     alpha=0,
    ##     beta=1,
    ##     norm_type=cv.NORM_MINMAX
    ## )

    # convert images from grayscale to RGB
    raw_img = cv.cvtColor(raw_img, cv.COLOR_GRAY2RGB)
    class_img = cv.cvtColor(class_img, cv.COLOR_GRAY2RGB)

    # change white areas in class image to red
    class_img = cv.merge([
        cv.split(class_img)[2],
        np.zeros(np.shape(class_img)[0:2]).astype(np.float32),
        np.zeros(np.shape(class_img)[0:2]).astype(np.float32)
    ])

    #combine the raw image and class image and save as png
    overlayed_image = cv.addWeighted(raw_img, 1, class_img, 0.5, 0)

    return overlayed_image

