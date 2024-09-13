#date: 2024-09-13T16:55:37Z
#url: https://api.github.com/gists/47a1ab9c2cc6fb1e0bb321bd70a77e86
#owner: https://api.github.com/users/tabedzki

# test_logging.py

import time
from pathlib import Path
import pprint
import logging
import warnings
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG) # Set the bug level of the package to be DEBUG by default. Allow user to override


def run_kilosort(settings, results_dir=None, filename=None):

    results_dir = Path(results_dir)
    # setup_logger_current(results_dir)
    setup_logger_proposed(results_dir)
    try:
        logger.debug("Debug message")
        logger.info(f"Kilosort version 4")
        logger.info(f"Sorting {filename}")
        logger.info('-'*40)
        will_fail()
    except Exception as e:
        # This makes sure the full traceback is written to log file.
        logger.exception('Encountered error in `run_kilosort`:')
        e.add_note(f'NOTE: See {results_dir}/kilosort4.log for detailed info.')
        raise e

def will_fail():
    0/0

def setup_logger_current(results_dir):
    # Adapted from
    # https://docs.python.org/2/howto/logging-cookbook.html#logging-to-multiple-destinations
    # In summary: only send logging.debug statements to log file, not console.

    # set up logging to file for root logger
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                        datefmt='%m-%d %H:%M',
                        filename=results_dir/'kilosort4.log',
                        filemode='w')

    # define a Handler which writes INFO messages or higher to the sys.stderr
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    # set a format which is simpler for console use
    console_formatter = logging.Formatter('%(name)-12s: %(message)s')
    console.setFormatter(console_formatter)
    # add the console handler to the root logger
    logging.getLogger('').addHandler(console)

    # Set 3rd party loggers to INFO or above only,
    # so that it doesn't spam the log file
    numba_log = logging.getLogger('numba')
    numba_log.setLevel(logging.INFO)

    mpl_log = logging.getLogger('matplotlib')
    mpl_log.setLevel(logging.INFO)

def setup_logger_proposed(results_dir):
    # Adapted from
    # https://docs.python.org/3/howto/logging.html#configuring-logging-for-a-library

    # Create handlers
    file_handler = logging.FileHandler(results_dir / 'kilosort4.log', mode='w')
    file_handler.setLevel(logging.DEBUG)

    # Create formatters and add them to the handlers
    file_formatter = logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s', datefmt='%m-%d %H:%M')
    file_handler.setFormatter(file_formatter)

    # Add handlers to the logger
    logger.addHandler(file_handler)

    # User should define log level for 3rd party applications in their own code.