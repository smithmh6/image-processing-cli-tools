"""
This module contains the build_model() function when
building a new model is required.
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

# global variables for drawing ROI's
DRAWING = False
NAMED_WINDOW = "Select ROI"
IX, IY = -1,-1
START_POINT = Point(-1, -1)
END_POINT = Point(-1, -1)
OPEN_IMG = None
TARGET_ROI = []
NON_TARGET_ROI = []


def draw_rectangle(event, x, y, flags, param):
    """
    Rectangular mouse callback function. Draws target
    roi's with left mouse button, non-target roi's with
    right mouse button.
    """

    # initialize global variables
    global IX, IY, DRAWING, START_POINT, END_POINT, TARGET_ROI, NON_TARGET_ROI

    if event == cv.EVENT_LBUTTONDOWN or event == cv.EVENT_RBUTTONDOWN:
        # get the starting state of the rectangle region
        DRAWING = True
        IX, IY = x, y
        START_POINT = Point(IX, IY)

    elif event == cv.EVENT_MOUSEMOVE and DRAWING:
        # render the image while drawing
        tmp = OPEN_IMG.copy()
        cv.rectangle(tmp, (IX, IY), (x,y), (0, 255, 255), 2)
        cv.imshow(NAMED_WINDOW, tmp)

    elif event == cv.EVENT_LBUTTONUP or event == cv.EVENT_RBUTTONUP:
        # handle the drawn region when done
        DRAWING = False
        draw_color = (0, 255, 0) if event == cv.EVENT_LBUTTONUP else (0, 0, 255)
        cv.rectangle(OPEN_IMG, (IX, IY), (x,y), draw_color, 2)
        END_POINT = Point(x, y)
        cv.imshow(NAMED_WINDOW, OPEN_IMG)

        if event == cv.EVENT_LBUTTONUP:
            TARGET_ROI.append(Rectangle(START_POINT, END_POINT))
            out_string = f'TARGET [{len(TARGET_ROI)}]: Start{START_POINT} ---> End{END_POINT}'
            sys.stdout.write(out_string + '\n')
            logging.info(out_string)

        elif event == cv.EVENT_RBUTTONUP:
            NON_TARGET_ROI.append(Rectangle(START_POINT, END_POINT))
            out_string = f'NON-TARGET [{len(NON_TARGET_ROI)}]: Start{START_POINT} ---> End{END_POINT}'
            sys.stdout.write(out_string + '\n')
            logging.info(out_string)


def draw_roi(img_dict:np.ndarray, open_im:np.ndarray) -> None:
    """
    Opens ND filter image in color to allow
    ROI's to be drawn with the mouse.
    """

    # initialize global variables
    global OPEN_IMG, NAMED_WINDOW

    # normalize the image
    OPEN_IMG = cv.normalize(
        open_im.astype(np.float32),
        dst=None,
        alpha=0,
        beta=1,
        norm_type=cv.NORM_MINMAX
    )
    # convert to color for drawing ROI's
    OPEN_IMG = cv.cvtColor(OPEN_IMG, cv.COLOR_GRAY2RGB)

    # open a named window to draw ROI's
    cv.namedWindow(NAMED_WINDOW)
    cv.setMouseCallback(NAMED_WINDOW, draw_rectangle)
    cv.startWindowThread()
    sys.stdout.write("Drawing ROI's (press ESC to exit):\n")
    while 1:
        cv.imshow(NAMED_WINDOW, OPEN_IMG)
        if cv.waitKey() & 0xFF == 27:
            break
        cv.destroyWindow(NAMED_WINDOW)


def build_model(cfg:ConfigParser):
    """
    Constructs an MOE prediction model from
    the selected dataset.
    """

    # initialize globals
    global TARGET_ROI, NON_TARGET_ROI, OPEN_IMG

    # store the images
    imgs = get_images(cfg)

    # calculate approximate memory usage of images
    img_mem = np.sum([sys.getsizeof(v) for v in imgs.values()])
    logging.info(f"Allocated {img_mem} bytes for image data.")

    # get the plot variables
    plot_vars = ast.literal_eval(cfg['output']['plot_vars'])
    logging.info(f"Found plot variables ---> {plot_vars}.")

    # create an hdf5 file to store results
    out_file_path = os.path.join(
        cfg['paths']['out'],
        cfg['paths']['run_label'],
        f"{cfg['paths']['run_label']}_model.h5"
    )
    logging.info(f"Creating output file at --> {out_file_path}")
    out_file = h5py.File(out_file_path, 'w')

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
        # verify ND filter is in list
        nd_found = False
        for k in imgs.keys():
            if not 'MOE' in k:
                nd_found = True
                break

        if not nd_found:
            raise ValueError("Neutral density filter not found. Cannot use Ratio mode.")

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

    # load existing ROI if requested, else draw a new one
    if cfg.getboolean('model', 'use_roi'):
        # open the data file (hdf5 file)
        logging.info(
            f"Reading {os.path.getsize(cfg['model']['model_path'])} \
                bytes from ROI model {cfg['model']['model_path']}.")

        # read the existing model file in readonly mode
        with h5py.File(cfg['model']['model_path'], 'r') as roi_model:

            # Parse the ROI's
            # write roi's to output file
            logging.info("Parsing ROI data.")

            # get the target regions
            targ_data = np.asarray(roi_model['roi']['target']['regions'])
            for reg in targ_data:
                # use the top left and bottom right corners
                # as points to make rectangle objects
                TARGET_ROI.append(
                    Rectangle(Point(reg[0][0], reg[0][1]), Point(reg[3][0], reg[3][1])))
            logging.info(f"Found {len(TARGET_ROI)} target regions.")
            logging.info(f"Targets ---> {TARGET_ROI}")

            # get the non-target regions
            non_targ_data = np.asarray(roi_model['roi']['non_target']['regions'])
            for reg in non_targ_data:
                # use the top left and bottom right corners
                # as points to make rectangle objects
                NON_TARGET_ROI.append(
                    Rectangle(Point(reg[0][0], reg[0][1]), Point(reg[3][0], reg[3][1])))
            logging.info(f"Found {len(NON_TARGET_ROI)} non-target regions.")
            logging.info(f"Non-Targets ---> {NON_TARGET_ROI}")

        # normalize the image
        OPEN_IMG = cv.normalize(
            list(imgs.values())[0].astype(np.float32),
            dst=None,
            alpha=0,
            beta=1,
            norm_type=cv.NORM_MINMAX
        )


        # convert to color for drawing ROI's
        OPEN_IMG = cv.cvtColor(OPEN_IMG, cv.COLOR_GRAY2RGB)

        for reg in TARGET_ROI:
            cv.rectangle(
                OPEN_IMG,
                (reg.point1.x, reg.point1.y),
                (reg.point2.x, reg.point2.y),
                (0, 255, 0), 2
            )

        for reg in NON_TARGET_ROI:
            cv.rectangle(
                OPEN_IMG,
                (reg.point1.x, reg.point1.y),
                (reg.point2.x, reg.point2.y),
                (0, 0, 255), 2
            )

        # save a plot of the ROI selections if requested
        if 'roi' in plot_vars:
            roi_fig = plt.figure(figsize=(8, 8))
            plt.imshow(cv.cvtColor(OPEN_IMG, cv.COLOR_RGB2BGR))
            roi_fig.savefig(os.path.join(
                cfg['paths']['out'], cfg['paths']['run_label'], f"{cfg['paths']['run_label']}_roi.png"))
            plt.close(roi_fig)

    else:
        draw_roi(imgs, list(imgs.values())[0])

        # write roi's to output file
        logging.info("Writing target ROI data to output file.")
        roi_grp = out_file.create_group("roi")
        targ_sub = roi_grp.create_group("target")
        targ_arr = np.array([list(r) for r in TARGET_ROI])
        targ_sub.create_dataset("regions", data=targ_arr, shape=targ_arr.shape, dtype=targ_arr.dtype)

        logging.info("Writing non-target ROI data to output file.")
        ntarg_sub = roi_grp.create_group("non_target")
        ntarg_arr = np.array([list(r) for r in NON_TARGET_ROI])
        ntarg_sub.create_dataset("regions", data=ntarg_arr, shape=ntarg_arr.shape, dtype=ntarg_arr.dtype)

        # save a plot of the ROI selections if requested
        if 'roi' in plot_vars:
            roi_fig = plt.figure(figsize=(8, 8))
            plt.imshow(cv.cvtColor(OPEN_IMG, cv.COLOR_RGB2BGR))
            roi_fig.savefig(os.path.join(
                cfg['paths']['out'], cfg['paths']['run_label'], f"{cfg['paths']['run_label']}_roi.png"))
            plt.close(roi_fig)

    # extract the target ROI's from images
    logging.info("Extracting target ROI intensity values.")
    targets = extract_roi(imgs, TARGET_ROI)

    # extract the non-target ROI's from images
    logging.info("Extracting non-target ROI intensity values.")
    non_targets = extract_roi(imgs, NON_TARGET_ROI)


    # generate model using requested fit function
    logging.info("Performing fit to the ROI data.")
    x_fit = np.hstack((targets, non_targets))
    y_fit = np.hstack((
        np.ones(targets[0, :].shape),
        np.zeros(non_targets[0, :].shape)
    ))

    fit_fx = fit_func(cfg.getint('model', 'fit_order'), len(imgs))
    opt, cov = curve_fit(fit_fx, x_fit, y_fit)

    logging.info("Writing model to output file.")
    model_grp = out_file.create_group("model")
    model_grp.attrs.create('fit_order', cfg.getint('model', 'fit_order'))
    model_grp.create_dataset("opt", data=opt, shape=opt.shape, dtype=opt.dtype)
    model_grp.create_dataset("cov", data=cov, shape=cov.shape, dtype=cov.dtype)

    # make the prediction using unpacking operator
    logging.info("Generating polynomial model.")
    prediction = fit_fx(x_fit, *opt)

    # write polynomial to output file
    logging.info("Writing output to file.")
    model_grp.create_dataset(
        "poly", data=prediction, shape=prediction.shape, dtype=prediction.dtype)

    # plot polynomial, if requested
    if 'poly' in plot_vars:
        logging.info("Plotting the polynomial function.")
        pred_fig = plt.figure(num="polynomial", figsize=(8, 6))
        plt.plot(prediction)
        plt.title("Polynomial Fit Function")
        plt.tight_layout()
        if cfg.getboolean('output', 'show_plots'):
            plt.show(block=False)
            plt.pause(0.1)

        # save the poly figure
        logging.info("Writing polynomial model to disk.")
        pred_fig.savefig(os.path.join(
            cfg['paths']['out'], cfg['paths']['run_label'], f"{cfg['paths']['run_label']}_poly.png"))


    # perform the ROC curve analysis
    logging.info("Performing ROC analysis.")
    roc = roc_curve(y_fit, prediction, np.arange(0, 1, 0.01))

    # write ROC results to output file
    logging.info("Writing ROC output to file.")
    roc_grp = out_file.create_group("roc")
    for k, v in roc.items():
        dat = np.asarray(v)
        roc_grp.create_dataset(k, data=dat, shape=dat.shape, dtype=dat.dtype)

    # plot the roc curve results, if requested
    if 'roc' in plot_vars:
        roc_plot = plt.figure(num="roc curve", figsize=(8, 6))
        plt.plot(roc['Pfa'], roc['Pd'])
        plt.title(f"Receiver Operator Characteristic Curve")
        plt.xlabel("Probability of False Alarm")
        plt.ylabel("Probability of Detection")
        plt.xlim((0.0, 1.0))
        plt.ylim((0.0, 1.0))
        plt.text(0.75, 0.1, f"Thresh= {roc['t_val']}\nAUROC= {round(roc['AUROC'], 2)}")
        if cfg.getboolean('output', 'show_plots'):
            plt.show(block=False)
            plt.pause(0.1)

        # save the ROC figure
        logging.info("Saving ROC plot to disk.")
        roc_plot.savefig(os.path.join(
            cfg['paths']['out'], cfg['paths']['run_label'], f"{cfg['paths']['run_label']}_roc.png"))

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
    out_file.close()

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