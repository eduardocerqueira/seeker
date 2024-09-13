#date: 2024-09-13T16:55:37Z
#url: https://api.github.com/gists/47a1ab9c2cc6fb1e0bb321bd70a77e86
#owner: https://api.github.com/users/tabedzki

# testing_logging_levels.py

import logging
from test_logging import run_kilosort

# Create handlers
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler('my_custom.log', mode='w')
file_handler.setLevel(logging.INFO)

# Create formatter and add it to the handlers
formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# Get the outer logger
outer_logger = logging.getLogger('outer_logger')
outer_logger.setLevel(logging.DEBUG)
outer_logger.addHandler(console_handler)
outer_logger.addHandler(file_handler)

# Log a message before calling run_kilosort
outer_logger.debug('This is a debug message from the outer logger before calling run_kilosort.')
outer_logger.info('This is an info message from the outer logger before calling run_kilosort.')

# Set logigng levels
test_logging_logger = logging.getLogger('test_logging')
# test_logging_logger.setLevel(logging.DEBUG) # Should a user decide to override the logging level
test_logging_logger.propagate = False

# Call the function from test_logging.py
try:
    run_kilosort(settings={}, results_dir='.')
except:
    outer_logger.exception("Kilosort4 failed")

# Log a message after calling run_kilosort
outer_logger.debug('This is a debug message from the outer logger after calling run_kilosort.')
outer_logger.info('This is an info message from the outer logger after calling run_kilosort.')