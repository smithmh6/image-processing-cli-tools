"""
This module contains a script that applies a pre-existing
model generated with the 'run_processing.py' script.
"""

# import packages
import ast
import argparse as ap
import configparser as cp
import cv2 as cv
import numpy as np
from os import listdir, mkdir
from os.path import join, normpath, isdir, isfile, exists
from scipy.signal import convolve2d
from tqdm import tqdm
from utils import (
    dark_correct,
    search_data,
    MOE, NBP, VIS, SWIR,
    ratiometric_fx_0,
    ratiometric_fx_1,
    ratiometric_fx_2,
    ratiometric_fx_3,
    flatfield_calibration,
    color_overlay
)

###############################################################################

# create command line arg parser
_ARG_PARSER = ap.ArgumentParser(
    prog="apply_model.py",
    description='Apply an existing MOE or Narrow Bandpass detection model to a dataset.'
)

# add positional arguments to arg parser
_ARG_PARSER.add_argument('CONFIG_PATH', type=str)

###############################################################################

# define constants
DEBUG = True
SHOW_IMAGES = True
KERNEL = None
ERODE = None
DILATION = None
OVERLAY_COLOR = False

###############################################################################

def apply_model(
    root_path:str, camera:str, scene:str,
    filter_set:str, data_set:str, output_dir:str,
    path_to_model:str, func_choice:int, moe_choices,
    flatfield_enabled:bool
):
    """
    Applies a pre-existing model to a dataset.
    """

    save_score = None
    save_smoothed = None
    save_detection = None
    save_eroded = None
    save_dilated = None
    false_color_img = None

    # validate the root path
    if not isinstance(root_path, str):
        raise TypeError(f'"root_path" arg expected <str> but received {type(root_path)}.')
    root_path = normpath(root_path)
    if not isdir(root_path):
        raise IOError(f'Root path {root_path} is not valid.')

    # validate the model path
    model_path = normpath(path_to_model)
    if not isinstance(path_to_model, str):
        raise TypeError(f'"path_to_model" arg expected type <str> but received {type(path_to_model)}.')
    if not isfile(model_path):
        raise IOError(f'Model file {model_path} not found.')

    # validate the output path
    output_directory = normpath(output_dir)
    if not isinstance(output_dir, str):
        raise TypeError(f'"output_dir" arg expects <str> but received {type(output_dir)}.')
    if not exists(output_directory):
        mkdir(output_directory)


    data_paths = search_data(root_path, camera, scene, filter_set, data_set)

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


    if DEBUG:
        print(f'\n{len(set_images_gray)} set images found.')
        for im, arr in set_images_gray.items():
            print(f'\n\t.\\{im}')

    # perform dark subtraction
    if DEBUG:
        print(f'\nPerforming dark subtraction on set images..')
    dark_corrected_images = dark_correct(
        set_images_gray, show_progress=DEBUG
    )

    if flatfield_enabled:
        dark_ff_images = dark_correct(
            ff_images_gray, show_progress=DEBUG
        )
        ff_corrected_images = flatfield_calibration(
            dark_corrected_images, dark_ff_images, show_progress=DEBUG
        )
        dark_corrected_images = ff_corrected_images


    # load the polynomial model
    if DEBUG:
        print('\nLoading model...')

    poly_coefficients = None
    roc_threshold = None
    with open(model_path, 'r') as m:

        roc_threshold = m.readline().split(':')[1]
        roc_threshold = float(roc_threshold.strip())
        print('\nThreshold: ', roc_threshold)

        poly_coefficients = m.readline().replace('\n', '').strip()
        poly_coefficients = poly_coefficients.split(',')
        poly_coefficients = np.array([
            float(str(x).strip()) for x in poly_coefficients if x not in ['', ' ']
        ])
        print('\nCoefficients:\n')
        print(poly_coefficients)


    if filter_set.upper() == NBP:

        # calculate the bandpass ratio
        if DEBUG:
            print(f'\nCalculating bandpass ratio..')

        # Bandpass generates a single image
        bandpass_ratio_image = cv.divide(
            list(dark_corrected_images.values())[0].astype(np.float),
            list(dark_corrected_images.values())[1].astype(np.float)
        )


        if DEBUG:
            print('\nGenerating score image..')

        bp_score = None
        with np.errstate(invalid='ignore'):
            bp_score = np.polyval(poly_coefficients, bandpass_ratio_image.flatten())
            bp_score = np.reshape(bp_score, np.shape(bandpass_ratio_image))



        if KERNEL is not None:

            if DEBUG:
                print('\nSmoothing score images..')

            bp_smooth_score = convolve2d(bp_score, KERNEL, mode='same')
            save_smoothed = cv.normalize(
                bp_smooth_score, None,
                alpha = 0, beta = 255,
                norm_type = cv.NORM_MINMAX,
                dtype = cv.CV_32F).astype(np.uint16)
        else:
            bp_smooth_score = bp_score

        if DEBUG:
            print('\nApplying threshold..')

        bp_ret, bp_thresh = cv.threshold(
            bp_smooth_score, roc_threshold, 1, cv.THRESH_BINARY
        )

        if SHOW_IMAGES:
            while 1:
                cv.imshow(
                    'Bandpass Detection Image for {scene}',
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
                        'Eroded Detection Image for {scene}',
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
                            'Dilated Detection Image for {scene}',
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
                            'Dilated Detection Image for {scene}',
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

        moe_combination = list(moe_choices)
        moe_key_choices = [
            str(list(dark_corrected_images.keys())[i])
            for i in range(len(list(dark_corrected_images.keys())))
            if i in moe_combination
        ]
        #print("Choices: ", moe_choices)
        print("\nMOE Selections: ")
        for i, k in zip(moe_combination, moe_key_choices):
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
                moe_nd_images[k] = cv.divide(
                    v.astype(np.float64),
                    nd_filter.astype(np.float64)
                )

        if DEBUG:
            print('\nGenerating score image..')

        # perform a fit based on the function choice
        fit_function = None
        if func_choice == 0:
            fit_function = ratiometric_fx_0

        if func_choice == 1:
            fit_function = ratiometric_fx_1

        elif func_choice == 2:
            fit_function = ratiometric_fx_2

        elif func_choice == 3:
            fit_function = ratiometric_fx_3


        moe_nd_score = None
        with np.errstate(invalid='ignore'):
            ratio_func = np.vectorize(fit_function, excluded=['data'])


            if func_choice == 0:
                moe_nd_score = ratio_func(
                    data=np.array([
                        list(moe_nd_images.values())[0].flatten(),
                        list(moe_nd_images.values())[1].flatten()
                    ]),
                    a=poly_coefficients[0],
                    b=poly_coefficients[1],
                    c=poly_coefficients[2]
                )

            elif func_choice == 1:
                moe_nd_score = ratio_func(
                    data=np.array([
                        list(moe_nd_images.values())[0].flatten(),
                        list(moe_nd_images.values())[1].flatten(),
                        list(moe_nd_images.values())[2].flatten()
                    ]),
                    a=poly_coefficients[0],
                    b=poly_coefficients[1],
                    c=poly_coefficients[2],
                    d=poly_coefficients[3]
                )


            elif func_choice == 2:
                moe_nd_score = ratio_func(
                    data=np.array([
                        list(moe_nd_images.values())[0].flatten(),
                        list(moe_nd_images.values())[1].flatten(),
                        list(moe_nd_images.values())[2].flatten()
                    ]),
                    a=poly_coefficients[0],
                    b=poly_coefficients[1],
                    c=poly_coefficients[2],
                    d=poly_coefficients[3],
                    e=poly_coefficients[4],
                    f=poly_coefficients[5],
                    g=poly_coefficients[6]
                )

            elif func_choice == 3:
                moe_nd_score = ratio_func(
                    data=np.array([
                        list(moe_nd_images.values())[0].flatten(),
                        list(moe_nd_images.values())[1].flatten(),
                        list(moe_nd_images.values())[2].flatten()
                    ]),
                    a=poly_coefficients[0],
                    b=poly_coefficients[1],
                    c=poly_coefficients[2],
                    d=poly_coefficients[3],
                    e=poly_coefficients[4],
                    f=poly_coefficients[5],
                    g=poly_coefficients[6],
                    h=poly_coefficients[7],
                    i=poly_coefficients[8],
                    j=poly_coefficients[9]
                )

            moe_nd_score = np.reshape(moe_nd_score, np.shape(list(moe_nd_images.values())[0]))

        if KERNEL is not None:

            if DEBUG:
                print('\nSmoothing score images..')

            moe_nd_smooth = convolve2d(moe_nd_score, KERNEL, mode='same')
            save_smoothed = cv.normalize(
                moe_nd_smooth, None,
                alpha = 0, beta = 255,
                norm_type = cv.NORM_MINMAX,
                dtype = cv.CV_32F).astype(np.uint16)
        else:
            moe_nd_smooth = moe_nd_score


        if DEBUG:
            print('\nApplying threshold..')
        moe_nd_ret, moe_nd_thresh = cv.threshold(
            moe_nd_smooth, roc_threshold, 1, cv.THRESH_BINARY
        )

        if SHOW_IMAGES:
            while 1:
                cv.imshow(
                    f'Bandpass Detection Image for {scene}',
                    cv.resize(
                        moe_nd_thresh,
                        (np.shape(moe_nd_thresh)[1]//2, np.shape(moe_nd_thresh)[0]//2)
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
                        'Eroded Detection Image for {scene}',
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
                            'Dilated Detection Image for {scene}',
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
                            'Dilated Detection Image for {scene}',
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

    to_save = input('\nWould you like to save these results? (y/n)\n')

    if "y" == to_save.lower():

        print(f'\nSaving to: {output_directory} ..')

        save_dir = output_directory
        if not isdir(save_dir):
            mkdir(save_dir)

        # save the images
        score_path = join(save_dir, f'{camera}_{data_set}_{scene}_{filter_set}_score.tif')
        score_smooth_path = join(save_dir, f'{camera}_{data_set}_{scene}_{filter_set}_score_smooth.tif')
        detection_path = join(save_dir, f'{camera}_{data_set}_{scene}_{filter_set}_detection.tif')
        cv.imwrite(score_path, save_score*255)

        cv.imwrite(detection_path, save_detection*255)

        if save_smoothed is not None:
            score_smooth_path = join(save_dir, f'{camera}_{data_set}_{scene}_{filter_set}_score_smooth.tif')
            cv.imwrite(score_smooth_path, save_smoothed*255)

        if save_dilated is not None:
            dilation_path = join(save_dir, f'{camera}_{data_set}_{scene}_{filter_set}_dilated.tif')
            cv.imwrite(dilation_path, save_dilated*255)

        if save_eroded is not None:
            erosion_path = join(save_dir, f'{camera}_{data_set}_{scene}_{filter_set}_eroded.tif')
            cv.imwrite(erosion_path, save_eroded*255)

        if false_color_img is not None:
            false_color_path = join(save_dir, f'{camera}_{data_set}_{scene}_{filter_set}_overlay.tif')
            cv.imwrite(false_color_path, false_color_img)

    print('\nDone.')


if __name__ == "__main__":

    # parse the command line arguments
    args = _ARG_PARSER.parse_args()

    # parse config file
    config = cp.ConfigParser()
    config.read(normpath(args.CONFIG_PATH))

    # set constants with input flags
    DEBUG = config['flags']['debug'] == 'True'
    SHOW_IMAGES = config['flags']['show_images'] == 'True'
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

    enable_flatfield = ast.literal_eval(config['images']['flatfield'])

    # call the main process using
    # the command line inputs
    apply_model(
        config['paths']['data'],
        config['dataset']['cam'],
        config['dataset']['scene'],
        config['dataset']['filter_set'],
        config['dataset']['set_name'],
        config['paths']['output'],
        config['paths']['model'],
        int(config['fitting']['fit_function']),
        ast.literal_eval(config['fitting']['combination']),
        enable_flatfield
    )
