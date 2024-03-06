"""
This module contains a set of utility functions used
during the processing of DSTL image data. There is also
a custom error class to handle image exceptions.
"""

import ast
import cv2 as cv
from errors import ImageError
from os.path import join, normpath, isdir, isfile
from os import walk, listdir
import numpy as np
import shapes
import sys
from tqdm import tqdm

# ----- CONSTANTS ----- #

# camera types
SWIR = "SWIR"
VIS = "VIS"

# filter Types
MOE = "MOE"
NBP = "NBP"

# ----- END CONSTANTS ----- #




def search_data(root_path:str, camera:str, scene:str, filter_set:str, data_set:str) -> dict:
    """
    Search the data file tree for the correct
    dataset and images.

    Parameters
    -----------

    Returns
    -----------
    (dict) key-value pairs of the full path to each input argument.
    """

    # validate the root path arg first
    if not isinstance(root_path, str):
        raise TypeError(f'"root_path" arg expected <str> but received {type(root_path)}.')
    if not isdir(root_path):
        raise ValueError(f'{root_path} is not a valid path.')

    # dict to store path at each directory level
    data_paths = {'root':normpath(root_path)}

    # validate the camera arg
    if not isinstance(camera, str):
        raise TypeError(f'"camera" arg expected <str> but received {type(camera)}.')

    if not camera.upper() in (SWIR, VIS):
        raise ValueError(f'Invalid camera name. Expected "MOE" or "SWIR", received {camera}.')

    data_paths['camera'] = join(data_paths['root'], camera)
    if not isdir(data_paths['camera']):
        raise ValueError('{} not found.'.format(data_paths['camera']))


    # validate scene arg
    if not isinstance(scene, str):
        raise ValueError(f'"scene" arg expected type <str> but received {type(scene)}.')

    data_paths['scene'] = join(data_paths['camera'], scene)  # store the scene path
    if not isdir(data_paths['scene']):
        raise ValueError('Scene directory {} not found.'.format(data_paths['scene']))


    # validate the filter_set arg
    if not isinstance(filter_set, str):
        raise ValueError(f'"set_name" expected type <str> but received {type(filter_set)}.')

    if not filter_set.upper() in (NBP, MOE):
        raise NameError(f'{filter_set} is not valid. Expected <str> "MOE or "Narrow Bandpass".')

    filter_set = "Narrow Bandpass" if filter_set == NBP else MOE
    data_paths['filter'] = join(data_paths['scene'], filter_set)
    if not isdir(data_paths['filter']):
        raise ValueError('{} not found.'.format(data_paths['filter']))


    # validate set arg
    if not isinstance(data_set, str):
        raise TypeError(f'"set" arg expected <str> but received {type(data_set)}.')

    data_paths['set'] = join(data_paths['filter'], data_set)
    if not isdir(data_paths['set']):
        raise ValueError('Set directory {} not found.'.format(data_paths['set']))

    data_paths['flatfield'] = join(data_paths['filter'], 'flatfield')
    if not isdir(data_paths['flatfield']):
        print('Flatfield directory {} not found.'.format(data_paths['flatfield']))

    return data_paths

def load_roi(roi_path:str):
    """
    Load pre-existing ROI's in for analysis.
    """
    if not isinstance(roi_path, str):
        raise TypeError(f'"roi_path" expected type <str> but received {type(roi_path)}.')

    roi_path = normpath(roi_path)
    if not isfile(roi_path):
        raise IOError(f'ROI file {roi_path} not found.')

    with open(roi_path, 'r') as r:

        # read the lines from the file and clean up characters
        target_line = r.readline().split(':')[1].replace('\n', '').replace(' ', '')
        non_target_line = r.readline().split(':')[1].replace(' ', '')

        # evaluate the lines into lists
        target_line = ast.literal_eval(target_line)
        non_target_line = ast.literal_eval(non_target_line)

        # set the ROI's
        target_roi = shapes.Rectangle(
            shapes.Point(target_line[0][0], target_line[0][1]),
            shapes.Point(target_line[1][0], target_line[1][1])
        )

        non_target = [
            shapes.Rectangle(
                shapes.Point(box[0][0], box[0][1]),
                shapes.Point(box[1][0], box[1][1])
            ) for box in non_target_line
        ]

        return target_roi, non_target

def dark_correct(images:dict, show_progress=True) -> dict:
    """
    Performs a dark correction on an image or list of images
    by means of subtracting the dark image from the image
    in question.

    Parameters
    -----------------
    im (dict): a dictionary of image names(keys) and np.ndarrays.\n
    show_progress (bool): default = False. show/hide progress bars.

    Returns
    -----------------
    (dict) image_arrays: a new dictionary containing the dark subtracted
        images as numpy arrays with the original input key values.
    """

    # make sure images arg is a dict
    if not isinstance(images, dict):
        raise ImageError(f'"im" arg must be of type <dict>. Received {type(images)} instead.')

    # make sure images arg is not empty
    if not images:
        raise ImageError(f'"im" arg cannot be empty.')

    # store the dark image
    dark_img = None
    # make a deep copy of 'images'
    imgs_copy = {
        k: v for k, v in images.items()
    }

    # verify there is a dark field image
    for k in imgs_copy.keys():
        if "dark" in str(k).lower():
            dark_img = imgs_copy.pop(k)
            break

    # raise exception if no dark image found
    if dark_img is None:
        raise ImageError("No dark field image found in dataset.")

    dark_corrected_imgs = {}

    # read the images
    try:

        # read the input images
        for k, im in tqdm(imgs_copy.items(), disable=not show_progress):

            # perform dark correction
            dark_corrected_imgs[k] =  cv.subtract(im, dark_img)


    except Exception as e:
        print(e)
        sys.exit()


    return dark_corrected_imgs

def flatfield_calibration(img:dict, ff:dict, show_progress=True) -> dict:
    """
    Calculates the flatfield correction on dark-subtracted
    images.

    Parameters
    ------------
    img (dict): dict containing key-value pairs of dark-corrected
        images with image file names as keys and dark-corrected
        image arrays as values.
    ff (dict): dict containing key-value pairs of dark corrected
        flatfield images with file names as keys and dark-corrected
        image arrays as values.
    show_progress (bool): default = False. show/hide progress bars.

    Returns
    ------------
    (dict) ff_cal: flatfield calibrated dictionary where keys are
        the names of the non-flatfield images and the values are the
        numpy arrays generated during the calculation.
    """

    # validate input arg types
    if not isinstance(img, dict) or not isinstance(ff, dict):
        raise ImageError(f'"img" and "ff" args expected type <dict>. Received {type(img)} and {type(ff)} instead.')

    # validate dict sizes, must be equal
    if not len(img) == len(ff):
        raise ImageError(f'"img" and "ff" args must have the same length. Received {len(img)} and {len(ff)} instead.')

    # store the results of the flatfield calibration
    ff_cal = {}

    # iterate both dicts at the same time
    for (k_i, v_i), (k_f, v_f) in tqdm(zip(img.items(), ff.items()), total=len(img), disable=show_progress):

        # perform the flatfield calculation
        # and add to ff_cal
        ff_cal[str(k_i)] = cv.divide(v_i.astype(np.float), v_f.astype(np.float))

    return ff_cal

def ratio_calibration(img:dict, show_progress=True) -> dict:
    """
    Performs a ratio calculation of MOE / ND on the image data.

    Parameters
    -----------------
    img (dict): key-value pairs of image names and image arrays.
    show_progress (bool): default = False. show/hide progress bars.

    Returns
    -----------------
    (dict) ratio: key-value pairs of input image name and resulting arrays.
    """


    # verify img arg type
    if not isinstance(img, dict):
        raise ImageError(f'"img" arg expected type <dict> but received {type(img)} instead.')

    # verify there is a neutral density image
    valid_nd = None
    for key in img.keys():
        if "nd" in str(key).lower() or "ne" in str(key).lower():
            valid_nd = key
            break

    if not valid_nd:
        raise ImageError("Neutral density image not found in dataset.")

    # store calculation results
    ratio = {}

    # iterate input data
    for k, v in tqdm(img.items(), disable=show_progress):

        # skip the nd filter
        if not str(k).lower() == str(valid_nd).lower():
            ratio[k] = cv.divide(v.astype(np.float), img[valid_nd].astype(np.float))

    return ratio

def bandpass_calibration(img:dict) -> np.ndarray:
    """
    Calculates the dark-corrected bandpass ratio of BandPass 1 / Bandpass2.

    Parameters
    -----------
    img (dict): key-value pairs of image names and dark corrected np.ndarray types.
        Must only contain 2 images.

    Returns
    -----------
    (np.ndarray) bp_cal: the results of the bandpass calculation.
    """

    # validate input arg
    if not isinstance(img, dict):
        raise ImageError(f'"img" arg expected type <dict> but received {type(img)}.')

    if not len(img) == 2:
        raise ImageError(f'"img" expected len() == 2. Received {len(img)}.')

    img_list = list(img.values())

    # perform the bandpass ratio calculation
    return cv.divide(img_list[0].astype(np.float), img_list[1].astype(np.float))

def bandpass_fx_1(x, a, b):
    return (a * x) + b

def bandpass_fx_2(x, a, b, c):
    return (a * x**2) + (b * x) + c

def ratiometric_fx_0(data, a, b, c):
    return (a * data[0]) + (b * data[1]) + c

def ratiometric_fx_1(data, a, b, c, d):
    return (a * data[0]) + (b * data[1]) + (c * data[2]) + d

def ratiometric_fx_2(data, a, b, c, d, e, f, g):
    return (
        (a * data[0]**2) + (b * data[0]) +
        (c * data[1]**2) + (d * data[1]) +
        (e * data[2]**2) + (f * data[2]) + g
    )

def ratiometric_fx_3(data, a, b, c, d, e, f, g, h, i, j):
    return (
        (a * data[0]**2) + (b * data[0])
        + (c * data[1]**2) + (d * data[1])
        + (e * data[2]**2) + (f * data[2])
        + (g * data[0] *data[1]) + (h * data[0] * data[2]) + (i * data[1] * data[2]) + j
    )

def _apply_flatfield_coefficients(img:dict) -> np.ndarray:
    raise NotImplementedError("Function _apply_flatfield_coefficients not implemented.")

def _apply_ratio_coefficients(img:dict) -> np.ndarray:
    raise NotImplementedError("Function _apply_ratio_coefficients not implemented.")

def _apply_bandpass_coefficients(img:dict) -> np.ndarray:
    raise NotImplementedError("Function _apply_bandpass_coefficients not implemented.")

def apply_coefficients(img:dict, mode:int) -> np.ndarray:
    """
    Applies the calibration coefficients to the processed images. Works
    like a factory method by using the mode arg to select the appropriate
    function used to apply the calibration coefficients.

    Parameters
    --------------
    img (dict): key-value pairs of images with image file names as keys
        and numpy.ndarray types as values. Assumes pre-processing.
    mode (int): determines which calibration calculation to be used.
        Must be in range 1, 2, or 3.

    Returns
    -------------
    (np.ndarray)  numpy.ndarray containing the output array
        of the calibration calculation. Output matches shape of input array's.

    Raises
    ------------
    ImageError() if input types are incorrect or mode is invalid.
    """

    # validate input values
    if not isinstance(img, dict):
        raise ImageError(f'"img" arg expected type <dict> but received {type(img)}.')

    if not isinstance(mode, int):
        raise ImageError(f'"mode" arg expect type <int> but received {type(mode)}.')

    if mode not in [1, 2, 3]:
        raise ImageError(f'"mode" value out of range. Expected 1, 2, or 3 but received {mode}.')


    if mode == 1:
        return _apply_flatfield_coefficients(img)

    elif mode == 2:
        return _apply_ratio_coefficients(img)

    elif mode == 3:
        return _apply_bandpass_coefficients(img)

    return np.array([0,0,0,0,0])

def generate_class_score():
    """
    """

    raise NotImplementedError("Function generate_class_score() is not implemented.")

def roc_curve(truth:np.ndarray, detections:np.ndarray, thresh:np.ndarray) -> dict:
    """
    Processes a set of detections, ground truth values and a threshold array
    into a ROC curve.

    Parameters
    -----------
    truth (array): logical index or binary values indicating class membership
            detections - measured results (or prediction scores) from sensor\n
    detections (array): measured results (or prediction scores) from sensor\n
    thresh (array): array of user specified threshold values for computing the ROC curve

    Returns
    ---------
    dictionary object of transmission and reflection value arrays\n
    {key : value}\n
    { 'AUROC' : Area Under the Receiver Operator Curve,\n
    'Pd' : Probability of detection (or sensitivity),\n
    'Pfa' : Probability of false alarm (or 1 - specificity),\n
    't_ind' : index of optimal threshold,\n
    't_val' : Optimal threshold based on distance to origin,\n
    'Se' : Optimal sensitivity based upon optimal threshold,\n
    'Sp' : Optimal specificity based upon optimal threshold }
    """

    # define a dictionary containing function variable names
    # Detections | True Positives | True Negatives | False Positives
    # | False Negatives | Probability of detection |
    # Probability of false alarm
    roc_vals = {'detects':np.zeros((1, np.shape(truth)[1])),
                'true_pos':np.zeros((1, len(thresh))),
                'true_neg':np.zeros((1, len(thresh))),
                'false_pos':np.zeros((1, len(thresh))),
                'false_neg':np.zeros((1, len(thresh))),
                'prob_det':np.zeros((1, len(thresh))),
                'prob_fa':np.zeros((1, len(thresh)))}

    # Run loop to threshold detections data and calculate TP, TN, FP & FN
    for i, val in enumerate(thresh):
        roc_vals['detects'] = np.array([
            1 if d >= val else 0 for d in np.squeeze(detections)
        ])
        roc_vals['true_pos'][:, i] = np.sum(truth * roc_vals['detects'])
        roc_vals['true_neg'][:, i] = (
            np.sum((1 - truth) * (1 - roc_vals['detects']))
        )
        roc_vals['false_pos'][:, i] = (
            np.sum(roc_vals['detects']) - roc_vals['true_pos'][:, i]
        )
        roc_vals['false_neg'][:, i] = (
            np.sum(truth) - roc_vals['true_pos'][:, i]
        )

    # Calculate Pd and Pfa
    roc_vals['prob_det'] = (
        roc_vals['true_pos']
        / (roc_vals['true_pos'] + roc_vals['false_neg'])
    )
    roc_vals['prob_fa'] = (
        1 - (
            roc_vals['true_neg']
            / (roc_vals['false_pos'] + roc_vals['true_neg'])
        )
    )

    # Calculate the optimal threshold
    # Define a vector of multiple indices of the ROC curve origin (upperleft)
    orig = np.tile([0, 1], (np.shape(roc_vals['prob_fa'])[1], 1))

    # Calculate the Euclidian distance from each point to the origin
    euc_dist = np.sqrt(np.square(orig[:, 0] - np.squeeze(roc_vals['prob_fa'].T))
        + np.square(orig[:, 1] - np.squeeze(roc_vals['prob_det'].T)))

    # Find the best threshold index
    t_ind = np.argmin(euc_dist)

    # Find the best threshold value
    t_value = thresh[t_ind]

    # Calculate the optimal sensitivity and specificity
    sensitivity = roc_vals['prob_det'][0, t_ind]
    specificity = 1 - roc_vals['prob_fa'][0, t_ind]

    # Calculate the AUROC using a simple summation of rectangles
    au_roc = -np.trapz(
        np.insert(roc_vals['prob_det'],
            np.shape(roc_vals['prob_det'])[1], 0, axis=1
        ),
        np.insert(roc_vals['prob_fa'], 0, 1, axis=1)
    )

    return {'AUROC':au_roc, 'Pd':roc_vals['prob_det'], 'Pfa':roc_vals['prob_fa'],
            't_val':t_value, 'Se':sensitivity, 'Sp':specificity}

def color_overlay(raw_img:np.ndarray, class_img:np.ndarray) -> np.ndarray:
    """
    Overlays raw image with class image.

    Parameters
    ----------
    raw_image_path (str): file path of the raw image as a 16-bit .tif file with dimensions 2448 x 2048. \n
    class_image_path (str): file path of the class image as a 16-bit .tif file. \n
    output_path (str): file path of the overlayed image saved as a .png file.

    Returns
    ----------
    overlayed_image (ndarray): an numpy array containing RGB values for the overlayed image.
    """

    # open raw image and rescale 12-bit integers to 16-bit integers
    #raw_img = cv.imread(raw_image_path, cv.IMREAD_UNCHANGED)
    #raw_img = ((2**16) - 1) / ((2**12) - 1) * raw_img
    #raw_img = raw_img.astype(np.uint16)

    # define the dimesnions of the raw image
    #height,width = raw_img.shape
    #dim = (width, height)

    # open class image and resize it to the same dimensions as the raw image
    #class_img = cv.imread(class_image_path, cv.IMREAD_UNCHANGED)
    #class_img = cv.resize(class_img, dim)

    # convert images from grayscale to RGB
    raw_img = cv.cvtColor(raw_img, cv.COLOR_GRAY2RGB)
    class_img = cv.cvtColor(class_img, cv.COLOR_GRAY2RGB)

    # change white areas in class image to red
    class_img = cv.merge([
        np.zeros(np.shape(class_img)[0:2]).astype(np.uint16),
        np.zeros(np.shape(class_img)[0:2]).astype(np.uint16),
        cv.split(class_img)[2]
    ])

    #combine the raw image and class image and save as png
    overlayed_image = cv.addWeighted(raw_img, 1, class_img, 0.5, 0)
    #cv.imwrite(output_path, overlayed_image)

    return overlayed_image




