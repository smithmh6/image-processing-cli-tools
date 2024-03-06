"""
The main processing script for USTER data.

Comments
------------
To execute, run the following command from the top-level
directory (place double quotes around config path, as shown):

>>> python -m scripts.main -c "<path to config>"

To execute directly from command shell:
>>> uster -c "<path to config>"
"""

# import dependencies
import argparse as ap
import configparser as cp
import logging
import os
import pathlib
from uster_processing.apply import apply_model
from uster_processing.build import build_model
import sys
import time


def main():
    """
    The main entry-point for USTER data processing.
    """

    # construct the Arg Parser
    ARGS = ap.ArgumentParser(prog="main.py", description="Main script for USTER data.")
    ARGS.add_argument("-c", "--config", type=pathlib.Path)

    # read the command line arg
    sys.stdout.write('Parsing input args..\n')
    args = ARGS.parse_args()

    # start timer and write to console
    sys.stdout.write('Starting main function..\n')
    start_time = time.perf_counter()

    # read the config file
    sys.stdout.write('Reading config file..\n')
    cfg = cp.ConfigParser()
    cfg.read(args.config)

    # check path to .h5 data
    sys.stdout.write(f"Checking data path ---> {cfg['paths']['data']}.\n")
    if not os.path.exists(cfg['paths']['data']):
        raise IOError(f"Data file path not found ---> {cfg['paths']['data']}")

    # define the output path for the current run
    run_path = os.path.join(cfg['paths']['out'], cfg['paths']['run_label'])

    # if overwrite == False, verify run path does not exist
    sys.stdout.write(f"Checking output path {run_path}.\n")
    if os.path.exists(run_path) and not cfg.getboolean('output', 'overwrite'):
        raise IOError(f"Cannot write to {run_path}. Path already exists.")

    # create the run path
    sys.stdout.write("Creating output path.\n")
    if not os.path.exists(run_path):
        os.mkdir(run_path)

    # set up the logger
    sys.stdout.write("Setting up log file.\n")
    logging.basicConfig(
        format='%(asctime)s %(message)s',
        datefmt='%m/%d/%Y %I:%M:%S %p',
        filename=os.path.join(run_path, f"{cfg['paths']['run_label']}.log"),
        level=logging.INFO,
        filemode='w'  # overwrite existing file
    )

    # determine wether we should build a new
    # model or apply an existing model
    if cfg.getboolean('model', 'build'):
        # check if use_roi is True, validate path
        if cfg.getboolean('model', 'use_roi'):
            if os.path.exists(cfg['model']['model_path']):
                sys.stdout.write(f"Building model with mode ---> {cfg['model']['mode']}.\n")
                sys.stdout.write(f"Using ROI found at ---> {cfg['model']['model_path']}.\n")
                build_model(cfg)
            else:
                raise IOError(f"ROI file path not found ---> {cfg['model']['model_path']}")
        else:
            sys.stdout.write(f"Building model with mode ---> {cfg['model']['mode']}.\n")
            build_model(cfg)

    elif not cfg.getboolean('model', 'build'):

        # check the path of the model name
        sys.stdout.write(f"Checking model path ---> {cfg['model']['model_path']}.\n")
        if os.path.exists(cfg['model']['model_path']):
            sys.stdout.write(f"Applying model using mode ---> {cfg['model']['mode']}.\n")
            apply_model(cfg)

        else:
            raise IOError(f"Model path not found ---> {cfg['model']['model_path']}")

    # end counter and output process time
    end_time = time.perf_counter()
    sys.stdout.write(f'Finished in {round((end_time - start_time), 3)} seconds.')
    logging.info(f'Finished in {round((end_time - start_time), 3)} seconds.')

    # close out stdout writer
    sys.stdout.close()

if __name__ == "__main__":

    # execute the main function
    # if called directly
    main()

