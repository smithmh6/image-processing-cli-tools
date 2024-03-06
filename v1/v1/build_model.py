"""
Entry point to CLI that handles batch processing
of DSTL image data.
"""

# import packages
import argparse as ap
import ast
import configparser as cp
import cv2 as cv
from itertools import combinations
import numpy as np
from os import listdir, mkdir
from os.path import join, normpath, isdir, isfile, exists
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
from scipy.signal import convolve2d
import shapes
import sys
from tqdm import tqdm
from utils import (
     MOE, NBP, SWIR, VIS,
    dark_correct,
    search_data,
    bandpass_fx_1,
    bandpass_fx_2, roc_curve,
    ratiometric_fx_0,
    ratiometric_fx_1,
    ratiometric_fx_2,
    ratiometric_fx_3,
    flatfield_calibration,
    color_overlay
)

###############################################################################
# create an arg parser
ARG_PARSER = ap.ArgumentParser(
    prog='build_model.py',
    description='Construct a prediction model from image data.'
)

# add an argument to input the config file path
ARG_PARSER.add_argument('CONFIG_PATH', type=str)

###############################################################################

# define constants
DEBUG = False
PLOTTING = False
SHOW_IMAGES = False
LOAD_ROI = False
KERNEL = None
ERODE = None
DILATION = None
OVERLAY_COLOR = False

###############################################################################

# define globals for drawing ROI's
_DRAWING = False
_IX, _IY = -1,-1
_START_POINT = shapes.Point(-1, -1)
_END_POINT = shapes.Point(-1, -1)
_OPEN_IMG = None
_TARGET_ROI = []
_NON_TARGET_ROI = []

###############################################################################

def draw_rectangle(event, x, y, flags, param):
    """
    Rectangular mouse callback function.
    """
    global _IX, _IY, _DRAWING, _START_POINT, _END_POINT, _TARGET_ROI, _NON_TARGET_ROI

    if event == cv.EVENT_LBUTTONDOWN:
        _DRAWING = True
        _IX, _IY = x, y

        _START_POINT = shapes.Point(_IX, _IY)
        if DEBUG:
            print(f'Start Point: {_START_POINT}')

    elif event == cv.EVENT_MOUSEMOVE and _DRAWING:
        tmp = _OPEN_IMG.copy()
        cv.rectangle(tmp, (_IX, _IY), (x,y), (0, 255, 255), 2)
        cv.imshow('Select ROI', tmp)


    elif event == cv.EVENT_LBUTTONUP:
        _DRAWING = False
        clr = (0, 255, 0) if not _TARGET_ROI else (0, 0, 255)
        cv.rectangle(_OPEN_IMG, (_IX, _IY), (x,y), clr, 2)
        cv.imshow('Select ROI', _OPEN_IMG)

        _END_POINT = shapes.Point(x, y)
        _ROI = shapes.Rectangle(_START_POINT, _END_POINT)

        if not _TARGET_ROI:
            _TARGET_ROI = shapes.Rectangle(_START_POINT, _END_POINT)

        elif _TARGET_ROI:
            _NON_TARGET_ROI.append(shapes.Rectangle(_START_POINT, _END_POINT))

        if DEBUG:
            print(f'End Point: {_END_POINT}')
            print(f'ROI: {_ROI}')

def process_main(
    root_path:str, camera:str, scene:str,
    filter_set:str, data_set:str, output_dir:str,
    roi_path:str, moe_choices, func_choice:int,
    flatfield_enabled:bool):
    """
    The main script used to calibrate DSTL image data.

    Parameters
    ----------------
    scene_path (str): full file path to scene folder.\n
    set_name (str): name of the set folder to be analyzed.\n
    proc_mode (int): 1 = MOE, 2 = NBP.\n
    moe_type (int): MOE type: 1 = Single Beam, 2 = Ratio. Unused if mode == 2.

    Returns
    ----------------
      (float) a single value with the prediction score.

    Raises
    ----------------
    ValueError() if any parameters are of incorrect types.
    NameError() if any file paths are not valid.
    ImageError() if any image-related issues arise.
    """

    global _OPEN_IMG, _NON_TARGET_ROI, _TARGET_ROI

    # define values used during file saving
    save_polynomial = None
    save_coefficients = None
    save_threshold = None
    save_roc = None
    save_score = None
    save_score_smooth = None
    save_detection = None
    save_function_text = None
    save_eroded = None
    save_dilated = None
    false_color_img = None

    # validate output directory
    output_directory = normpath(output_dir)
    if not exists(output_directory):
        mkdir(output_directory)

    # pass input args to search_data() function
    data_paths = search_data(root_path, camera, scene, filter_set, data_set)

    # collect the set images
    set_rgb_images = {}
    for s in listdir(data_paths['set']):
        if ".tif" in str(s).lower() or ".png" in str(s).lower():
            if camera == VIS:
                set_rgb_images[str(s)] = cv.resize(
                    cv.imread(join(data_paths['set'], str(s))), (1224, 1024)
                )
            elif camera == SWIR:
                set_rgb_images[str(s)] = cv.imread(join(data_paths['set'], str(s)))

    # read the set images in as grayscale
    set_images_gray = {}
    for s in listdir(data_paths['set']):
        if "set_details" not in str(s).lower() and "scene_image" not in str(s).lower():

            temp = cv.imread(join(data_paths['set'], str(s)), -1)
            temp = temp.astype(np.float64) / 65535.0

            # resize images if VIS
            if camera == VIS:
                h, w = temp.shape
                set_images_gray[str(s)] = cv.resize(temp, (w//2, h//2))
            elif camera == SWIR:
                set_images_gray[str(s)] = temp

    if DEBUG:
        print(f'\n{len(set_images_gray)} set images found.')
        for im, arr in set_images_gray.items():
            print(f'\t.\\{im}')

    if flatfield_enabled:
        print("\nLoading flatfield images...")

        # read the set images in as grayscale
        ff_images_gray = {}
        for s in listdir(data_paths['flatfield']):
            if "set_details" not in str(s).lower() and "scene_image" not in str(s).lower():

                temp = cv.imread(join(data_paths['flatfield'], str(s)), -1)
                temp = temp.astype(np.float64) / 65535.0

                # resize images if VIS
                if camera == VIS:
                    h, w = temp.shape
                    ff_images_gray[str(s)] = cv.resize(temp, (w//2, h//2))
                elif camera == SWIR:
                    ff_images_gray[str(s)] = temp


    # if LOAD_ROI is not active, show selection image
    if not LOAD_ROI:

        # display an image from the set to choose ROI:
        if DEBUG:
            print("\nDrawing ROI's..")

        _OPEN_IMG = cv.normalize(
            list(set_rgb_images.values())[1],
            dst=None, alpha=0, beta=255,
            norm_type=cv.NORM_MINMAX
        )
        cv.namedWindow("Select ROI")
        cv.setMouseCallback("Select ROI", draw_rectangle)

        while 1:
            cv.imshow(
                "Select ROI",
                _OPEN_IMG
            )
            if cv.waitKey() & 0xFF == 27:
                break
        cv.destroyAllWindows()

        # display the roi selections
        if DEBUG:
            print("\nTarget ROI: ", _TARGET_ROI)
            print("\nNon-Target ROIs: ", _NON_TARGET_ROI)

    elif LOAD_ROI:

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
            _TARGET_ROI = shapes.Rectangle(
                shapes.Point(target_line[0][0], target_line[0][1]),
                shapes.Point(target_line[1][0], target_line[1][1])
            )

            _NON_TARGET_ROI = [
                shapes.Rectangle(
                    shapes.Point(box[0][0], box[0][1]),
                    shapes.Point(box[1][0], box[1][1])
                ) for box in non_target_line
            ]

            print('\nTarget ROI: ', _TARGET_ROI)
            print('\nNon-Target ROIs: ', _NON_TARGET_ROI)

            # show the display image with loaded roi's
            _OPEN_IMG = cv.normalize(
                list(set_rgb_images.values())[1],
                dst=None, alpha=0, beta=255,
                norm_type=cv.NORM_MINMAX
            )

            # draw the target ROI
            cv.rectangle(
                _OPEN_IMG,
                (_TARGET_ROI.point1.x, _TARGET_ROI.point1.y),
                (_TARGET_ROI.point2.x, _TARGET_ROI.point2.y),
                (0, 255, 0), 2
            )

            # draw the non-target roi's
            for box in _NON_TARGET_ROI:
                cv.rectangle(
                    _OPEN_IMG,
                    (box.top_left.x, box.top_left.y),
                    (box.bottom_right.x, box.bottom_right.y),
                    (0, 0, 255), 2
                )
            while 1:
                cv.imshow(
                    "Imported ROIs",
                    _OPEN_IMG
                )
                if cv.waitKey() & 0xFF == 27:
                    break
            cv.destroyAllWindows()


    # perform dark subtraction
    if DEBUG:
        print(f'\nPerforming dark subtraction on set images..')
    dark_corrected_images = dark_correct(
        set_images_gray, show_progress=DEBUG
    )

    if flatfield_enabled:
        # perform dark subtraction
        if DEBUG:
            print(f'\nPerforming flatfield correction on set images..')

        dark_ff_images = dark_correct(
            ff_images_gray, show_progress=DEBUG
        )
        ff_corrected_images = flatfield_calibration(
            dark_corrected_images, dark_ff_images, show_progress=DEBUG
        )
        dark_corrected_images = ff_corrected_images

    # ratio the narrow bandpass images
    if filter_set.upper() == NBP:

        if DEBUG:
            print(f'\nCalculating bandpass ratio..')

        # Bandpass generates a single image
        bandpass_ratio_image = cv.divide(
            list(dark_corrected_images.values())[0].astype(np.float),
            list(dark_corrected_images.values())[1].astype(np.float)
        )

        if DEBUG:
            print(f'\nExtracting ROI data..')

        # remove nans and infs from target class data
        temp_1 = (
            bandpass_ratio_image[
                _TARGET_ROI.top_left.y : _TARGET_ROI.bottom_left.y+1,
                _TARGET_ROI.top_left.x : _TARGET_ROI.top_right.x+1
            ]
        ).flatten()

        temp_1_nans = np.argwhere(np.isnan(temp_1))
        temp_1_rm_nans = np.delete(temp_1, temp_1_nans)

        temp_1_infs = np.argwhere(np.isinf(temp_1_rm_nans))
        bp_target_class = np.delete(temp_1_rm_nans, temp_1_infs)


        # remove nans and infs from non-target class data
        temp_2 = np.concatenate([
            (bandpass_ratio_image[
                _NON_TARGET_ROI[i].top_left.y : _NON_TARGET_ROI[i].bottom_left.y+1,
                _NON_TARGET_ROI[i].top_left.x : _NON_TARGET_ROI[i].top_right.x+1
            ]).flatten() for i in range(len(_NON_TARGET_ROI))
        ])

        temp_2_nans = np.argwhere(np.isnan(temp_2))
        temp_2_rm_nans = np.delete(temp_2, temp_2_nans)

        temp_2_infs = np.argwhere(np.isinf(temp_2_rm_nans))
        bp_non_target_class = np.delete(temp_2_rm_nans, temp_2_infs)

        bp_x_fit_vals = np.concatenate((bp_target_class, bp_non_target_class), axis=0)
        bp_y_fit_vals = np.concatenate((
            np.ones(np.shape(bp_target_class)),
            np.zeros(np.shape(bp_non_target_class))
        ), axis=0)

        if DEBUG:
            print('\nGenerating the model...')

        if func_choice == 1:
            fit_function = bandpass_fx_1
            save_function_text = "Fit Function: a*x + b"

        elif func_choice == 2:
            fit_function = bandpass_fx_2
            save_function_text = "Fit Function: a*x^2 + b*x + c"

        bp_opt, bp_cov = curve_fit(
            fit_function, bp_x_fit_vals, bp_y_fit_vals
        )

        bp_predict = np.polyval(bp_opt, bp_x_fit_vals)

        save_coefficients = bp_opt

        save_polynomial = plt.figure()
        ax = plt.subplot(111)
        ax.plot(bp_predict)
        plt.title(f'Bandpass Model {scene} - {data_set}')

        if PLOTTING:
            plt.show()


        if DEBUG:
            print('\nCalculating the ROC curve..')

        bp_roc = roc_curve(
            np.reshape(bp_y_fit_vals, (1, len(bp_y_fit_vals))),
            np.reshape(bp_predict, (1, len(bp_predict))),
            np.arange(0, 1, 0.01)
        )
        save_threshold = bp_roc['t_val']

        save_roc = plt.figure()
        ax = plt.subplot(111)
        ax.plot(bp_roc['Pfa'][0], bp_roc['Pd'][0])
        plt.title(f'ROC Curve {scene} - {data_set} : Threshold = {bp_roc["t_val"]}')

        if PLOTTING:
            print('\nROC Curve output: ', bp_roc)
            plt.show()

        if DEBUG:
            print('\nGenerating score image..')

        bp_score = None
        with np.errstate(invalid='ignore'):
            bp_score = np.polyval(bp_opt, bandpass_ratio_image.flatten())
            bp_score = np.reshape(bp_score, np.shape(bandpass_ratio_image))

        if KERNEL is not None:

            if DEBUG:
                print('\nSmoothing score images..')

            bp_smooth_score = convolve2d(bp_score, KERNEL, mode='same')
            save_score_smooth = cv.normalize(
                bp_smooth_score, None,
                alpha = 0, beta = 255,
                norm_type = cv.NORM_MINMAX,
                dtype = cv.CV_32F).astype(np.uint16)
        else:
            bp_smooth_score = bp_score

        if DEBUG:
            print('\nApplying threshold..')

        bp_ret, bp_thresh = cv.threshold(bp_smooth_score, bp_roc['t_val'], 1, cv.THRESH_BINARY)

        if SHOW_IMAGES:
            while 1:
                cv.imshow(
                    f'Bandpass Detection Image for {scene}',
                    cv.resize(
                        bp_thresh,
                        (np.shape(bp_thresh)[1]//2, np.shape(bp_thresh)[0]//2)
                    )
                )
                if cv.waitKey() & 0xFF == 27:
                    break
            cv.destroyAllWindows()

        if ERODE is not None:

            if DEBUG:
                print('\nApplying erosion..')

            bp_erosion = cv.erode(bp_thresh, ERODE)
            save_eroded = cv.normalize(
                bp_erosion, None,
                alpha = 0, beta = 255,
                norm_type = cv.NORM_MINMAX,
                dtype = cv.CV_32F).astype(np.uint16)

            if SHOW_IMAGES:
                while 1:
                    cv.imshow(
                        f'Eroded Detection Image for {scene}',
                        cv.resize(
                            bp_erosion,
                            (np.shape(bp_erosion)[1]//2, np.shape(bp_erosion)[0]//2)
                        )
                    )
                    if cv.waitKey() & 0xFF == 27:
                        break
                cv.destroyAllWindows()

            if DILATION is not None:

                if DEBUG:
                    print('\nApplying dilation..')

                bp_dilation = cv.dilate(bp_erosion, DILATION)
                save_dilated = cv.normalize(
                    bp_dilation, None,
                    alpha = 0, beta = 255,
                    norm_type = cv.NORM_MINMAX,
                    dtype = cv.CV_32F).astype(np.uint16)


                if SHOW_IMAGES:
                    while 1:
                        cv.imshow(
                            f'Dilated Detection Image for {scene}',
                            cv.resize(
                                bp_dilation,
                                (np.shape(bp_dilation)[1]//2, np.shape(bp_dilation)[0]//2)
                            )
                        )
                        if cv.waitKey() & 0xFF == 27:
                            break
                    cv.destroyAllWindows()

        elif ERODE is None:

            if DILATION is not None:

                if DEBUG:
                    print('\nApplying dilation..')

                bp_dilation = cv.dilate(bp_thresh, DILATION)
                save_dilated = cv.normalize(
                    bp_dilation, None,
                    alpha = 0, beta = 255,
                    norm_type = cv.NORM_MINMAX,
                    dtype = cv.CV_32F).astype(np.uint16)


                if SHOW_IMAGES:
                    while 1:
                        cv.imshow(
                            f'Dilated Detection Image for {scene}',
                            cv.resize(
                                bp_dilation,
                                (np.shape(bp_dilation)[1]//2, np.shape(bp_dilation)[0]//2)
                            )
                        )
                        if cv.waitKey() & 0xFF == 27:
                            break
                    cv.destroyAllWindows()

        save_score = cv.normalize(
            bp_score, None,
            alpha = 0, beta = 255,
            norm_type = cv.NORM_MINMAX,
            dtype = cv.CV_32F).astype(np.uint16)
        save_detection = cv.normalize(
            bp_thresh, None,
            alpha = 0, beta = 255,
            norm_type = cv.NORM_MINMAX,
            dtype = cv.CV_32F).astype(np.uint16)


        if OVERLAY_COLOR:
            if camera == VIS:
                raw_overlay = (list(set_images_gray.values())[1] * 2**16).astype(np.uint16) * 16

            elif camera == SWIR:
                raw_overlay = (list(set_images_gray.values())[1] * 2**16).astype(np.uint16)

            false_color_img = color_overlay(raw_overlay, (save_detection * 257).astype(np.uint16))

            if SHOW_IMAGES:
                while 1:
                    cv.imshow(f'Color Overlay Image for {scene}', false_color_img)
                    if cv.waitKey() & 0xFF == 27:
                        break
                cv.destroyAllWindows()


    elif filter_set.upper() == MOE:

        # this is used if option 'all' is used for moe choices
        roc_details = []

        for moe_tuple in moe_choices:

            moe_key_choices = [
                str(list(dark_corrected_images.keys())[i])
                for i in range(len(list(dark_corrected_images.keys())))
                if i in moe_tuple
            ]

            print("\nMOE Selections: ")
            for i, k in zip(moe_tuple, moe_key_choices):
                print(f'[{i}]\t{k}')


            if DEBUG:
                print(f'\nCalculating MOE/ND ratios..')

            # MOE generates an image for each MOE filter
            moe_nd_images = {}
            nd_filter = None
            for k in dark_corrected_images.keys():
                if "NE" in str(k).upper():
                    nd_filter = dark_corrected_images[k]

            for k, v in tqdm(dark_corrected_images.items(), disable=not DEBUG):
                if str(k) in moe_key_choices:
                    moe_nd_images[k] = cv.divide(v, nd_filter)

            if DEBUG:
                print('\nExtracting ROI data..')

            moe_nd_target_class = []
            moe_nd_non_target_class = []

            for img in moe_nd_images.values():

                # remove nans and inf's from flattened arrays before appending
                temp_1 = (
                    img[
                        _TARGET_ROI.top_left.y : _TARGET_ROI.bottom_left.y+1,
                        _TARGET_ROI.top_left.x : _TARGET_ROI.top_right.x+1
                    ]
                ).flatten()

                temp_1_nans = np.argwhere(np.isnan(temp_1))
                temp_1_rm_nans = np.delete(temp_1, temp_1_nans)

                temp_1_infs = np.argwhere(np.isinf(temp_1_rm_nans))
                temp_1_rm_infs = np.delete(temp_1_rm_nans, temp_1_infs)

                moe_nd_target_class.append(temp_1_rm_infs)

                # remove nans and inf's from flattened arrays before appending
                temp_2 = np.concatenate([
                    (img[
                        _NON_TARGET_ROI[i].top_left.y : _NON_TARGET_ROI[i].bottom_left.y+1,
                        _NON_TARGET_ROI[i].top_left.x : _NON_TARGET_ROI[i].top_right.x+1
                    ]).flatten() for i in range(len(_NON_TARGET_ROI))
                ])
                temp_2_nans = np.argwhere(np.isnan(temp_2))
                temp_2_rm_nans = np.delete(temp_2, temp_2_nans)

                temp_2_infs = np.argwhere(np.isinf(temp_2_rm_nans))
                temp_2_rm_infs = np.delete(temp_2_rm_nans, temp_2_infs)

                moe_nd_non_target_class.append(temp_2_rm_infs)


            moe_nd_target_class = np.array(moe_nd_target_class)
            moe_nd_non_target_class = np.array(moe_nd_non_target_class)

            moe_nd_x_fit_vals = np.hstack((moe_nd_target_class, moe_nd_non_target_class))
            moe_nd_y_fit_vals = np.hstack((
                np.ones(np.shape(moe_nd_target_class[0])),
                np.zeros(np.shape(moe_nd_non_target_class[0]))
            ))

            if DEBUG:
                print('\nGenerating the model...')

            # perform a fit based on the function choice
            fit_function = None
            if func_choice == 0:
                fit_function = ratiometric_fx_0
                save_function_text = "Fit Function: a*x + b*y + c"
            elif func_choice == 1:
                fit_function = ratiometric_fx_1
                save_function_text = "Fit Function: a*x + b*y + c*z + d"

            elif func_choice == 2:
                fit_function = ratiometric_fx_2
                save_function_text = "Fit Function: a*x^2 + b*x + c*y^2 + d*y + e*z^2 + f*z + g"

            elif func_choice == 3:
                fit_function = ratiometric_fx_3
                save_function_text = "Fit Function: a*x^2 + b*x + c*y^2 + d*y + e*z^2 + f*Z + g*x*y + h*x*z + i*y*z + j"


            moe_nd_opt, moe_nd_cov = curve_fit(
                fit_function, moe_nd_x_fit_vals, moe_nd_y_fit_vals
            )

            save_coefficients = moe_nd_opt

            if func_choice == 0:
                moe_nd_predict = np.array([
                    fit_function(
                        [moe_nd_x_fit_vals[0][i], moe_nd_x_fit_vals[1][i]],
                        moe_nd_opt[0],
                        moe_nd_opt[1],
                        moe_nd_opt[2]
                    ) for i in range(np.shape(moe_nd_x_fit_vals)[1])
                ])

            elif func_choice == 1:
                moe_nd_predict = np.array([
                    fit_function(
                        [moe_nd_x_fit_vals[0][i], moe_nd_x_fit_vals[1][i], moe_nd_x_fit_vals[2][i]],
                        moe_nd_opt[0],
                        moe_nd_opt[1],
                        moe_nd_opt[2],
                        moe_nd_opt[3]
                    ) for i in range(np.shape(moe_nd_x_fit_vals)[1])
                ])

            elif func_choice == 2:
                moe_nd_predict = np.array([
                    fit_function(
                        [moe_nd_x_fit_vals[0][i], moe_nd_x_fit_vals[1][i], moe_nd_x_fit_vals[2][i]],
                        moe_nd_opt[0],
                        moe_nd_opt[1],
                        moe_nd_opt[2],
                        moe_nd_opt[3],
                        moe_nd_opt[4],
                        moe_nd_opt[5],
                        moe_nd_opt[6]
                    ) for i in range(np.shape(moe_nd_x_fit_vals)[1])
                ])

            elif func_choice == 3:
                moe_nd_predict = np.array([
                    fit_function(
                        [moe_nd_x_fit_vals[0][i], moe_nd_x_fit_vals[1][i], moe_nd_x_fit_vals[2][i]],
                        moe_nd_opt[0],
                        moe_nd_opt[1],
                        moe_nd_opt[2],
                        moe_nd_opt[3],
                        moe_nd_opt[4],
                        moe_nd_opt[5],
                        moe_nd_opt[6],
                        moe_nd_opt[7],
                        moe_nd_opt[8],
                        moe_nd_opt[9]
                    ) for i in range(np.shape(moe_nd_x_fit_vals)[1])
                ])



            save_polynomial = plt.figure()
            ax = plt.subplot(111)
            ax.plot(moe_nd_predict)
            plt.title(f'MOE Model {scene} - {data_set}')

            if PLOTTING:
                plt.show()

            if DEBUG:
                print('\nCalculating the ROC curve..')

            moe_nd_roc = roc_curve(
                np.reshape(moe_nd_y_fit_vals, (1, len(moe_nd_y_fit_vals))),
                np.reshape(moe_nd_predict, (1, len(moe_nd_predict))),
                np.arange(0, 1, 0.01)
            )

            save_threshold = moe_nd_roc['t_val']

            save_roc = plt.figure()
            ax = plt.subplot(111)
            ax.plot(moe_nd_roc['Pfa'][0], moe_nd_roc['Pd'][0])
            plt.title(f'ROC Curve {scene} - {data_set} : Threshold = {moe_nd_roc["t_val"]}')

            if PLOTTING:
                print('\nROC Curve output: ', moe_nd_roc)
                plt.show()

            if DEBUG:
                print('\nGenerating score image..')

            moe_nd_score = None
            with np.errstate(invalid='ignore'):
                ratio_func = np.vectorize(fit_function, excluded=['data'])

                if func_choice == 0:
                    moe_nd_score = ratio_func(
                        data=np.array([
                            list(moe_nd_images.values())[0].flatten(),
                            list(moe_nd_images.values())[1].flatten()
                        ]),
                        a=moe_nd_opt[0],
                        b=moe_nd_opt[1],
                        c=moe_nd_opt[2]
                    )


                elif func_choice == 1:
                    moe_nd_score = ratio_func(
                        data=np.array([
                            list(moe_nd_images.values())[0].flatten(),
                            list(moe_nd_images.values())[1].flatten(),
                            list(moe_nd_images.values())[2].flatten()
                        ]),
                        a=moe_nd_opt[0],
                        b=moe_nd_opt[1],
                        c=moe_nd_opt[2],
                        d=moe_nd_opt[3]
                    )


                elif func_choice == 2:
                    moe_nd_score = ratio_func(
                        data=np.array([
                            list(moe_nd_images.values())[0].flatten(),
                            list(moe_nd_images.values())[1].flatten(),
                            list(moe_nd_images.values())[2].flatten()
                        ]),
                        a=moe_nd_opt[0],
                        b=moe_nd_opt[1],
                        c=moe_nd_opt[2],
                        d=moe_nd_opt[3],
                        e=moe_nd_opt[4],
                        f=moe_nd_opt[5],
                        g=moe_nd_opt[6]
                    )

                elif func_choice == 3:
                    moe_nd_score = ratio_func(
                        data=np.array([
                            list(moe_nd_images.values())[0].flatten(),
                            list(moe_nd_images.values())[1].flatten(),
                            list(moe_nd_images.values())[2].flatten()
                        ]),
                        a=moe_nd_opt[0],
                        b=moe_nd_opt[1],
                        c=moe_nd_opt[2],
                        d=moe_nd_opt[3],
                        e=moe_nd_opt[4],
                        f=moe_nd_opt[5],
                        g=moe_nd_opt[6],
                        h=moe_nd_opt[7],
                        i=moe_nd_opt[8],
                        j=moe_nd_opt[9]
                    )


                moe_nd_score = np.reshape(moe_nd_score, np.shape(list(moe_nd_images.values())[0]))


            if KERNEL is not None:

                if DEBUG:
                    print('\nSmoothing score images..')

                moe_nd_smooth = convolve2d(moe_nd_score, KERNEL, mode='same')
                save_score_smooth = cv.normalize(
                    moe_nd_smooth, None,
                    alpha = 0, beta = 255,
                    norm_type = cv.NORM_MINMAX,
                    dtype = cv.CV_32F).astype(np.uint16)
            else:
                moe_nd_smooth = moe_nd_score


            if DEBUG:
                print('\nApplying threshold..')
            moe_nd_ret, moe_nd_thresh = cv.threshold(
                moe_nd_smooth, moe_nd_roc['t_val'], 1, cv.THRESH_BINARY
            )

            if SHOW_IMAGES:
                while 1:
                    cv.imshow(
                        f'MOE Detection Image for {scene}',
                        cv.resize(
                            moe_nd_thresh,
                            (np.shape(moe_nd_thresh)[1], np.shape(moe_nd_thresh)[0])
                        )
                    )
                    if cv.waitKey() & 0xFF == 27:
                        break
                cv.destroyAllWindows()


            if ERODE is not None:


                if DEBUG:
                    print('\nApplying erosion..')

                moe_nd_erosion = cv.erode(moe_nd_thresh, ERODE)
                save_eroded = cv.normalize(
                    moe_nd_erosion, None,
                    alpha = 0, beta = 255,
                    norm_type = cv.NORM_MINMAX,
                    dtype = cv.CV_32F).astype(np.uint16)

                if SHOW_IMAGES:
                    while 1:
                        cv.imshow(
                            f'Eroded Detection Image for {scene}',
                            cv.resize(
                                moe_nd_erosion,
                                (np.shape(moe_nd_erosion)[1]//2, np.shape(moe_nd_erosion)[0]//2)
                            )
                        )
                        if cv.waitKey() & 0xFF == 27:
                            break
                    cv.destroyAllWindows()

                if DILATION is not None:

                    if DEBUG:
                        print('\nApplying dilation..')

                    moe_nd_dilation = cv.dilate(moe_nd_erosion, DILATION)
                    save_dilated = cv.normalize(
                        moe_nd_dilation, None,
                        alpha = 0, beta = 255,
                        norm_type = cv.NORM_MINMAX,
                        dtype = cv.CV_32F).astype(np.uint16)


                    if SHOW_IMAGES:
                        while 1:
                            cv.imshow(
                                f'Dilated Detection Image for {scene}',
                                cv.resize(
                                    moe_nd_dilation,
                                    (np.shape(moe_nd_dilation)[1]//2, np.shape(moe_nd_dilation)[0]//2)
                                )
                            )
                            if cv.waitKey() & 0xFF == 27:
                                break
                        cv.destroyAllWindows()

            elif ERODE is None:

                if DILATION is not None:

                    if DEBUG:
                        print('\nApplying dilation..')

                    moe_nd_dilation = cv.dilate(moe_nd_thresh, DILATION)
                    save_dilated = cv.normalize(
                        moe_nd_dilation, None,
                        alpha = 0, beta = 255,
                        norm_type = cv.NORM_MINMAX,
                        dtype = cv.CV_32F).astype(np.uint16)


                    if SHOW_IMAGES:
                        while 1:
                            cv.imshow(
                                f'Dilated Detection Image for {scene}',
                                cv.resize(
                                    moe_nd_dilation,
                                    (np.shape(moe_nd_dilation)[1]//2, np.shape(moe_nd_dilation)[0]//2)
                                )
                            )
                            if cv.waitKey() & 0xFF == 27:
                                break
                        cv.destroyAllWindows()

            save_score = cv.normalize(
                moe_nd_score, None,
                alpha = 0, beta = 255,
                norm_type = cv.NORM_MINMAX,
                dtype = cv.CV_32F).astype(np.uint16)
            save_detection = cv.normalize(
                moe_nd_thresh, None,
                alpha = 0, beta = 255,
                norm_type = cv.NORM_MINMAX,
                dtype = cv.CV_32F).astype(np.uint16)

            if OVERLAY_COLOR:
                if camera == VIS:
                    raw_overlay = (list(set_images_gray.values())[1] * 2**16).astype(np.uint16) * 16

                elif camera == SWIR:
                    raw_overlay = (list(set_images_gray.values())[1] * 2**16).astype(np.uint16)

                false_color_img = color_overlay(raw_overlay, (save_detection * 257).astype(np.uint16))

                if SHOW_IMAGES:
                    while 1:
                        cv.imshow(f'Color Overlay Image for {scene}', false_color_img)
                        if cv.waitKey() & 0xFF == 27:
                            break
                    cv.destroyAllWindows()

            # extract the target array from the detection image
            targ_arr = (moe_nd_thresh[
                            _TARGET_ROI.top_left.y : _TARGET_ROI.bottom_left.y+1,
                            _TARGET_ROI.top_left.x : _TARGET_ROI.top_right.x+1
                        ]
                    ).flatten()

            # compute the number of false detection pixels in detection image
            false_detect_ratio = (np.count_nonzero(moe_nd_thresh) - len(targ_arr))/len(moe_nd_thresh.flatten())

            # append the information to the roc_details
            roc_details.append([moe_tuple, moe_nd_roc['t_val'], moe_nd_roc['AUROC'][0], false_detect_ratio])

            # save detection images and clear plots
            if len(moe_choices) > 1:

                # save the detection image
                moe_combo_string = f'{moe_tuple[0]}{moe_tuple[1]}{moe_tuple[2]}'
                detection_path = join(
                    output_directory, f'{moe_combo_string}_{camera}_{data_set}_{scene}_{filter_set}_detection.tif')
                cv.imwrite(detection_path, save_detection*255)



                # reset plt figure
                plt.clf(), plt.close()
                plt.clf(), plt.close()
                plt.clf(), plt.close()

    if len(moe_choices) == 1:

        to_save = input('\nWould you like to save these results? (y/n)\n')

        if "y" == to_save.lower():

            print(f'\nSaving to: {output_directory} ..')

            save_dir = output_directory
            if not isdir(save_dir):
                mkdir(save_dir)

            # save the plot models
            model_path = join(save_dir, f'{camera}_{data_set}_{scene}_{filter_set}_model.png')
            roc_path = join(save_dir, f'{camera}_{data_set}_{scene}_{filter_set}_roc.png')
            save_polynomial.savefig(model_path)
            save_roc.savefig(roc_path)

            # save the images
            score_path = join(save_dir, f'{camera}_{data_set}_{scene}_{filter_set}_score.tif')
            detection_path = join(save_dir, f'{camera}_{data_set}_{scene}_{filter_set}_detection.tif')
            roi_image_path = join(save_dir, f'{camera}_{data_set}_{scene}_{filter_set}_roi.tif')
            cv.imwrite(score_path, save_score*255)
            cv.imwrite(detection_path, save_detection*255)
            cv.imwrite(roi_image_path, _OPEN_IMG)

            if save_score_smooth is not None:
                score_smooth_path = join(save_dir, f'{camera}_{data_set}_{scene}_{filter_set}_score_smooth.tif')
                cv.imwrite(score_smooth_path, save_score_smooth*255)

            if save_dilated is not None:
                dilation_path = join(save_dir, f'{camera}_{data_set}_{scene}_{filter_set}_dilated.tif')
                cv.imwrite(dilation_path, save_dilated*255)

            if save_eroded is not None:
                erosion_path = join(save_dir, f'{camera}_{data_set}_{scene}_{filter_set}_eroded.tif')
                cv.imwrite(erosion_path, save_eroded*255)

            if false_color_img is not None:
                false_color_path = join(save_dir, f'{camera}_{data_set}_{scene}_{filter_set}_overlay.tif')
                cv.imwrite(false_color_path, false_color_img)

            # write the polynomial to a text file
            poly_path = join(save_dir, f'{camera}_{data_set}_{scene}_{filter_set}_poly.txt')
            with open(poly_path, 'w') as f:
                # write the ROC threshold
                f.write("Threshold: " + str(save_threshold) + "\n")
                for p in save_coefficients:

                    f.write(str(p) + ', ')

                f.write('\n\n' + save_function_text)

            roi_path = join(save_dir, f'{camera}_{data_set}_{scene}_{filter_set}_roi.txt')
            with open(roi_path, 'w') as r:
                # write roi's to file
                targ = [(_TARGET_ROI.point1.x, _TARGET_ROI.point1.y), (_TARGET_ROI.point2.x, _TARGET_ROI.point2.y)]
                r.write(f'Target: {targ}\n')
                non_targs = [
                    [
                        (_NON_TARGET_ROI[i].top_left.x , _NON_TARGET_ROI[i].top_left.y),
                        (_NON_TARGET_ROI[i].bottom_right.x , _NON_TARGET_ROI[i].bottom_right.y)
                    ]
                    for i in range(len(_NON_TARGET_ROI))
                ]

                r.write(f'Non-Targets: {non_targs}')

    elif len(moe_choices) > 1:
        roc_matrix = np.matrix(roc_details, dtype=object)
        cols = ['moes', 'thresh', 'auroc', 'false_detection']
        roc_df = pd.DataFrame(roc_matrix, columns=cols)
        roc_df.to_csv(
            join(
                output_directory,
                f'{camera}_{data_set}_{scene}_{filter_set}_roc_details.csv'
            ), sep=',', header=True, index=False
        )

    print('\nDone.')


if __name__ == "__main__":

    # parse command line args
    args = ARG_PARSER.parse_args()

    # parse config file
    config = cp.ConfigParser()
    config.read(normpath(args.CONFIG_PATH))

    # set constants with input flags
    DEBUG = config['flags']['debug'] == 'True'
    PLOTTING = config['flags']['show_plots'] == 'True'
    SHOW_IMAGES = config['flags']['show_images'] == 'True'
    LOAD_ROI = config['flags']['load_roi'] == 'True'
    OVERLAY_COLOR = config['flags']['false_color'] == 'True'


    # set the smoothing kernel
    img_settings = config['images']
    if int(img_settings['smooth']) > 0:
        KERNEL = np.ones(
            (int(img_settings['smooth']),
            int(img_settings['smooth'])),
            np.float32
        ) / int(img_settings['smooth'])**2

    # set the erosion kernel
    if int(img_settings['erode']) > 0:
        ERODE = np.ones(
            (int(img_settings['erode']),
            int(img_settings['erode'])),
            np.uint8
        )

    # set the dilation kernal
    if int(img_settings['dilate']) > 0:
        DILATION = np.ones(
            (int(img_settings['dilate']),
            int(img_settings['dilate'])),
            np.uint8
        )

    MOE_CHOICES = None
    if config['fitting']['combination'] == 'all':
        MOE_CHOICES = list(combinations([1, 2, 3, 4, 5, 6], 3))
    else:
        MOE_CHOICES = [ast.literal_eval(config['fitting']['combination'])]

    enable_flatfield = ast.literal_eval(config['images']['flatfield'])

    # call the main process using
    # command line inputs
    process_main(
        config['paths']['data'],
        config['dataset']['cam'],
        config['dataset']['scene'],
        config['dataset']['filter_set'],
        config['dataset']['set_name'],
        config['paths']['output'],
        config['paths']['roi'],
        MOE_CHOICES,
        int(config['fitting']['fit_function']),
        enable_flatfield
    )


